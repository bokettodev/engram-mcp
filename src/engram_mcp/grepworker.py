"""Bare stdlib-only child for bounded ``grep_index`` regex execution."""

import json
import re
import sys


def main():
    try:
        header = json.loads(sys.stdin.buffer.readline())
        rx = re.compile(str(header["pattern"]), int(header.get("flags", 0)))
        include_lines = bool(header.get("include_lines"))
        max_matches = int(header["max_matches"])
        by_path = {}
        total_matches = 0
        stopped = False
        for line in sys.stdin.buffer:
            batch = json.loads(line)
            for row in batch:
                content = row.get("content") or ""
                rel = row.get("rel_path") or ""
                start = int(row.get("start_line", 1) or 1)
                line_cache = None
                for match in rx.finditer(content):
                    item = by_path.setdefault(
                        rel,
                        {"path": rel, "match_count": 0, "line_numbers": set(), "lines": []},
                    )
                    total_matches += 1
                    item["match_count"] += 1
                    line_no = start + content[: match.start()].count("\n")
                    item["line_numbers"].add(line_no)
                    if include_lines:
                        if line_cache is None:
                            line_cache = content.splitlines()
                        idx = max(0, min(line_no - start, len(line_cache) - 1))
                        line_text = line_cache[idx] if line_cache else ""
                        item["lines"].append({"line": line_no, "text": line_text[:300]})
                    if total_matches >= max_matches:
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break
        for item in by_path.values():
            item["line_numbers"] = sorted(item["line_numbers"])
        result = {
            "status": "ok",
            "payload": {
                "by_path": by_path,
                "total_matches": total_matches,
                "stopped": stopped,
            },
        }
    except Exception as exc:
        result = {"status": "error", "error": str(exc) or repr(exc)}
    sys.stdout.write(json.dumps(result, ensure_ascii=False))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
