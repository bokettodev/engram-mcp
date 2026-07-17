"""CLI standard-stream configuration tests."""

from __future__ import annotations


def test_configure_standard_streams_uses_utf8(monkeypatch) -> None:
    from engram_mcp import cli

    class RecordingStream:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def reconfigure(self, **kwargs: str) -> None:
            self.calls.append(kwargs)

    stdout = RecordingStream()
    stderr = RecordingStream()
    monkeypatch.setattr(cli.sys, "stdout", stdout)
    monkeypatch.setattr(cli.sys, "stderr", stderr)

    cli._configure_standard_streams()

    expected = [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert stdout.calls == expected
    assert stderr.calls == expected
