from cantocaptions_ai.utils.log_utils import TranscriptionSummary


def test_a_stage_run_once_per_file_group_is_one_row():
    summary = TranscriptionSummary()
    summary.record("VAD", 1.0, 10.0, 300.0)
    summary.record("Transcription", 2.0, 60.0, 5800.0)
    summary.record("VAD", 1.5, 12.0, 280.0)
    summary.record("Transcription", None, 50.0, 6100.0)
    assert summary._stages == [
        ("VAD", 2.5, 22.0, 300.0),
        ("Transcription", 2.0, 110.0, 6100.0),
    ]


def test_stages_without_a_load_or_vram_stay_without_one():
    summary = TranscriptionSummary()
    summary.record("Speaker assignment", None, 0.1)
    summary.record("Speaker assignment", None, 0.2)
    ((label, load, run, vram),) = summary._stages
    assert (label, load, round(run, 3), vram) == ("Speaker assignment", None, 0.3, None)
