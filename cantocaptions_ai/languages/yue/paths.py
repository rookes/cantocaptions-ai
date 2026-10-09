"""Where the Cantonese pack's data files live."""
from pathlib import Path

PACK_DIR = Path(__file__).parent
RULES_DIR = PACK_DIR / "rules"      # cleaning manifests and regex rule files
OPENCC_DIR = PACK_DIR / "opencc"    # OpenCC configs for Simplified -> HK Traditional
PROOFREAD_DIR = PACK_DIR / "proofread"  # conventions for the LLM proofreading stage
