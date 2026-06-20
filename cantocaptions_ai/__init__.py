import importlib

def _lazy_import(name):
    module = importlib.import_module(f"cantocaptions_ai.{name}")
    return module

def load_align_model(*args, **kwargs):
    alignment = _lazy_import("pipeline.alignment")
    return alignment.load_align_model(*args, **kwargs)

def align(*args, **kwargs):
    alignment = _lazy_import("pipeline.alignment")
    return alignment.align(*args, **kwargs)

def load_model(*args, **kwargs):
    asr = _lazy_import("pipeline.asr")
    return asr.load_model(*args, **kwargs)

def load_audio(*args, **kwargs):
    audio = _lazy_import("utils.audio")
    return audio.load_audio(*args, **kwargs)

def assign_word_speakers(*args, **kwargs):
    diarize = _lazy_import("pipeline.diarize")
    return diarize.assign_word_speakers(*args, **kwargs)

def setup_logging(*args, **kwargs):
    logging_module = _lazy_import("utils.log_utils")
    return logging_module.setup_logging(*args, **kwargs)

def get_logger(*args, **kwargs):
    logging_module = _lazy_import("utils.log_utils")
    return logging_module.get_logger(*args, **kwargs)
