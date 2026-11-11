"""Komut satırından hızlı deneme:

    python -m src.guardrail.cli --source kaynak.txt --summary ozet.txt [--json]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .verifier import GuardrailVerifier

_ANSI = {"green": "\033[92m", "red": "\033[91m"}
_RESET = "\033[0m"


def main() -> None:
    ap = argparse.ArgumentParser(description="K4 NLI guardrail")
    ap.add_argument("--source", required=True, help="kaynak metin dosyası")
    ap.add_argument("--summary", required=True, help="özet dosyası")
    ap.add_argument("--json", action="store_true", help="JSON çıktı ver")
    args = ap.parse_args()

    source = Path(args.source).read_text(encoding="utf-8")
    summary = Path(args.summary).read_text(encoding="utf-8")
    report = GuardrailVerifier().verify(source, summary)

    if args.json:
        print(report.to_json())
        return
    for r in report.results:
        tag = r.verdict.value.upper()
        extra = f" [{r.reason}]" if r.reason else ""
        print(f"{_ANSI[r.ui_color]}[{tag} {r.confidence:.2f}]{_RESET} {r.sentence}{extra}")
    print(f"\nSadakat skoru: {report.faithfulness_score:.0%} "
          f"({len(report.hallucinated)}/{len(report.results)} cümle şüpheli)")


if __name__ == "__main__":
    main()
