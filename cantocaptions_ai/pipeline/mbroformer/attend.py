from functools import wraps
from packaging import version
from collections import namedtuple

import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce

try:  # torch >= 2.1
    from torch.nn.attention import SDPBackend as _SDPBackend, sdpa_kernel as _sdpa_kernel
except ImportError:  # pragma: no cover - older torch
    _SDPBackend = _sdpa_kernel = None

# constants

FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])


def _sdpa_backends(config: FlashAttentionConfig):
    """Context manager restricting SDPA to the backends `config` enables.

    torch.backends.cuda.sdp_kernel is deprecated (and warns on every call, which
    this enters depth x (time + freq) x 2 times per forward). Prefer the modern
    torch.nn.attention.sdpa_kernel where it exists.
    """
    if _sdpa_kernel is None:
        return torch.backends.cuda.sdp_kernel(**config._asdict())
    backends = []
    if config.enable_flash:
        backends.append(_SDPBackend.FLASH_ATTENTION)
    if config.enable_mem_efficient:
        backends.append(_SDPBackend.EFFICIENT_ATTENTION)
    if config.enable_math:
        backends.append(_SDPBackend.MATH)
    return _sdpa_kernel(backends)

# helpers

def exists(val):
    return val is not None

def once(fn):
    called = False
    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)

# main class

class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False
    ):
        super().__init__()
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and version.parse(torch.__version__) < version.parse('2.0.0')), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        # determine efficient attention configs for cuda and cpu

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        # Upstream gated flash on `major == 8 and minor == 0`, i.e. A100 only, so
        # every other SM -- including sm_86/89/90 consumer and Hopper parts -- fell
        # back to math + mem-efficient. Flash is supported on all of sm_80+.
        # Measured worth ~0% here (attention is not this model's bottleneck at
        # seq_len ~801), but there is no reason to keep asking for the slow kernel.
        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if device_properties.major >= 8:
            self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        _, heads, q_len, _, k_len, is_cuda, device = *q.shape, k.shape[-2], q.is_cuda, q.device

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with _sdpa_backends(config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p = self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        q_len, k_len, device = q.shape[-2], k.shape[-2], q.device

        scale = q.shape[-1] ** -0.5

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        return out
