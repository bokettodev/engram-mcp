"""Bare stdlib-only child for bounded caller-supplied regex execution."""

import json
import re
import sys


def _extract_first(rx, texts):
    out = []
    for text in texts:
        match = rx.search(text)
        if not match:
            out.append("")
        elif "ticket" in match.groupdict():
            out.append(match.group("ticket"))
        elif match.groups():
            out.append(match.group(1))
        else:
            out.append(match.group(0))
    return out


def main():
    try:
        job = json.loads(sys.stdin.buffer.read())
        rx = re.compile(str(job["pattern"]), int(job.get("flags", 0)))
        texts = [str(text or "") for text in job.get("texts", [])]
        op = job.get("op")
        if op == "search":
            payload = [bool(rx.search(text)) for text in texts]
        elif op == "extract":
            payload = _extract_first(rx, texts)
        else:
            raise ValueError(f"unsupported regex operation: {op!r}")
        result = {"status": "ok", "payload": payload}
    except Exception as exc:
        result = {"status": "error", "error": str(exc) or repr(exc)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
