"""Embedded subtitle tracks as reference subtitles: text tracks converted, bitmap tracks (PGS,
DVB, VobSub) read by OCR in a worker process. See docs/references.md.

Nothing is imported here eagerly: the OCR worker imports submodules of this package and must
not pull in the rest of the pipeline (or torch) by doing so.
"""


def extract_references(*args, **kwargs):
    from cantocaptions_ai.references.extract import extract_references as _extract
    return _extract(*args, **kwargs)
