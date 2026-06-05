import argparse
import sys
from pathlib import Path

from .io import pair, read_frequent_words, read_transcripts
from .report import write_error_stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="vosk-errs",
        description="Compute WER between reference and hypothesis transcripts "
        "and print a detailed error report.",
    )
    p.add_argument("ref", type=Path, help="reference file (utt_id words...)")
    p.add_argument("hyp", type=Path, help="hypothesis file (utt_id words...)")
    p.add_argument(
        "-i",
        "--ignore-case",
        action="store_true",
        help="lowercase tokens before comparing",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write the report to this file (default: stdout)",
    )
    p.add_argument(
        "-f",
        "--freq",
        type=Path,
        help="word frequency file (`<logprob>\\t<word>` per line); "
        "enables a rare-word substitution rate in the report",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=50000,
        help="words ranked in the top-N of --freq are treated as common; "
        "the rest are 'rare' (default: 50000)",
    )
    args = p.parse_args(argv)

    ref = read_transcripts(args.ref)
    hyp = read_transcripts(args.hyp)
    pairs, missing, extra = pair(ref, hyp)

    if missing:
        print(
            f"warning: {len(missing)} ref utterance(s) missing from hyp "
            f"(ignored): {', '.join(missing[:5])}"
            + ("..." if len(missing) > 5 else ""),
            file=sys.stderr,
        )
    if extra:
        print(
            f"warning: {len(extra)} hyp utterance(s) not in ref (ignored): "
            f"{', '.join(extra[:5])}" + ("..." if len(extra) > 5 else ""),
            file=sys.stderr,
        )

    rows = []
    for utt_id, r, h in pairs:
        if args.ignore_case:
            r = [w.lower() for w in r]
            h = [w.lower() for w in h]
        rows.append((utt_id, r, h))

    # Missing utterances (no hyp line) are ignored, not scored.

    frequent_words = None
    if args.freq is not None:
        frequent_words = read_frequent_words(
            args.freq, args.top_n, lowercase=args.ignore_case
        )

    if args.output is None:
        write_error_stats(
            sys.stdout, rows, frequent_words=frequent_words, top_n=args.top_n
        )
    else:
        with open(args.output, "w", encoding="utf-8") as fout:
            write_error_stats(
                fout, rows, frequent_words=frequent_words, top_n=args.top_n
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
