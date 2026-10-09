import pytest


@pytest.fixture(autouse=True)
def _no_default_log_file(monkeypatch):
    """The CLI writes a log under {output_dir}/logs by default, and tests that drive cli()
    mostly leave output_dir at its default -- the checkout's own output/. A test that wants
    a log file passes --log_file."""
    import cantocaptions_ai.__main__ as main

    resolve = main._resolve_log_file
    monkeypatch.setattr(
        main, "_resolve_log_file",
        lambda log_file, merged: resolve(log_file, merged) if log_file is not None else None,
    )
