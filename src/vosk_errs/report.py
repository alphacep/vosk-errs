"""icefall-compatible error report.

Mirrors the output of `icefall.utils.write_error_stats` so existing
analysis tooling can consume it. The sections are:

    %WER = ...
    Errors: ... insertions, ... deletions, ... substitutions, ...
    PER-UTT DETAILS: corr or (ref->hyp)
    SUBSTITUTIONS: count ref -> hyp
    DELETIONS: count ref
    INSERTIONS: count hyp
    PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp
"""

from collections import defaultdict
from typing import Iterable, List, Optional, Set, TextIO, Tuple

from .align import Op, align, score

ERR = "*"


def _pairs(ref, hyp):
    """Return kaldialign-style (ref_word, hyp_word) pairs, with `*` for gaps."""
    return [
        (r if r is not None else ERR, h if h is not None else ERR)
        for _, r, h in align(ref, hyp)
    ]


def _combine_successive_errors(ali):
    """Merge adjacent error cells, joining tokens with spaces.

    Replicates icefall's combine_successive_errors=True so a substitution
    next to a deletion renders as one `(a b->a)` cell instead of two.
    """
    ali = [[[x], [y]] for x, y in ali]
    for i in range(len(ali) - 1):
        if ali[i][0] != ali[i][1] and ali[i + 1][0] != ali[i + 1][1]:
            ali[i + 1][0] = ali[i][0] + ali[i + 1][0]
            ali[i + 1][1] = ali[i][1] + ali[i + 1][1]
            ali[i] = [[], []]
    ali = [
        [
            [t for t in x if t != ERR],
            [t for t in y if t != ERR],
        ]
        for x, y in ali
    ]
    ali = [cell for cell in ali if cell != [[], []]]
    return [
        (
            ERR if not x else " ".join(x),
            ERR if not y else " ".join(y),
        )
        for x, y in ali
    ]


def write_error_stats(
    f: TextIO,
    results: Iterable[Tuple[str, List[str], List[str]]],
    frequent_words: Optional[Set[str]] = None,
    rare_words: Optional[Set[str]] = None,
    top_n: Optional[int] = None,
) -> float:
    """Write an icefall-style detailed error report to `f`.

    `results` is an iterable of `(utt_id, ref_words, hyp_words)`. Returns
    the overall WER as a percentage rounded to 2 decimals (matches icefall).
    """
    if frequent_words is not None and rare_words is not None:
        raise ValueError("frequent_words and rare_words are mutually exclusive")

    results = list(results)

    subs: dict = defaultdict(int)
    rare_subs: dict = defaultdict(int)
    ins: dict = defaultdict(int)
    dels: dict = defaultdict(int)
    # corr, ref_sub, hyp_sub, ins, dels
    words: dict = defaultdict(lambda: [0, 0, 0, 0, 0])
    num_corr = 0

    rare_ref_count = 0
    rare_sub_count = 0
    report_rare_words = frequent_words is not None or rare_words is not None

    def is_rare(word: str) -> bool:
        if rare_words is not None:
            return word in rare_words
        if frequent_words is not None:
            return word not in frequent_words
        return False

    alis = []
    for utt_id, ref, hyp in results:
        ali = _pairs(ref, hyp)
        alis.append((utt_id, ali))
        for ref_word, hyp_word in ali:
            if ref_word == ERR:
                ins[hyp_word] += 1
                words[hyp_word][3] += 1
            elif hyp_word == ERR:
                dels[ref_word] += 1
                words[ref_word][4] += 1
            elif hyp_word != ref_word:
                subs[(ref_word, hyp_word)] += 1
                words[ref_word][1] += 1
                words[hyp_word][2] += 1
                if report_rare_words and is_rare(ref_word):
                    rare_subs[(ref_word, hyp_word)] += 1
                    rare_sub_count += 1
            else:
                words[ref_word][0] += 1
                num_corr += 1
            if (
                report_rare_words
                and ref_word != ERR
                and is_rare(ref_word)
            ):
                rare_ref_count += 1

    ref_len = sum(len(r) for _, r, _ in results)
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs
    tot_err_rate = "%.2f" % (100.0 * tot_errs / ref_len if ref_len else 0.0)

    char_errs = 0
    char_ref_len = 0
    for _, ref, hyp in results:
        ref_chars = list(" ".join(ref))
        hyp_chars = list(" ".join(hyp))
        s, _ = score(ref_chars, hyp_chars)
        char_errs += s.errors
        char_ref_len += s.ref_words
    cer = "%.2f" % (100.0 * char_errs / char_ref_len if char_ref_len else 0.0)

    rare_str = ""
    if report_rare_words:
        sub_rate = "%.2f" % (
            100.0 * sub_errs / ref_len if ref_len else 0.0
        )
        rare_rate = "%.2f" % (
            100.0 * rare_sub_count / rare_ref_count if rare_ref_count else 0.0
        )
        rare_str = "SRATE: " + \
                   f"{sub_rate} ({sub_errs}/{ref_len}) " + \
                   f"{rare_rate} ({rare_sub_count}/{rare_ref_count})"
    print(f"WER: {tot_err_rate}   CER: {cer}   I/D/S: {ins_errs}/{del_errs}/{sub_errs} ({num_corr}/{ref_len})   {rare_str}",
          file=f)
    print("", file=f)
    print("SUBSTITUTIONS: count ref -> hyp", file=f)
    for count, (r, h) in sorted(((v, k) for k, v in subs.items()), reverse=True):
        print(f"{count}   {r} -> {h}", file=f)

    if report_rare_words:
        print("", file=f)
        print("RARE WORD SUBSTITUTIONS: count ref -> hyp", file=f)
        for count, (r, h) in sorted(
            ((v, k) for k, v in rare_subs.items()), reverse=True
        ):
            print(f"{count}   {r} -> {h}", file=f)

    print("", file=f)
    print("DELETIONS: count ref", file=f)
    for count, r in sorted(((v, k) for k, v in dels.items()), reverse=True):
        print(f"{count}   {r}", file=f)

    print("", file=f)
    print("INSERTIONS: count hyp", file=f)
    for count, h in sorted(((v, k) for k, v in ins.items()), reverse=True):
        print(f"{count}   {h}", file=f)

    print("", file=f)
    print("PER-UTT DETAILS: corr or (ref->hyp)  ", file=f)
    for utt_id, ali in alis:
        merged = _combine_successive_errors(ali)
        rendered = " ".join(
            r if r == h else f"({r}->{h})" for r, h in merged
        )
        print(f"{utt_id}:\t{rendered}", file=f)

    print("", file=f)
    print(
        "PER-WORD STATS: word  corr tot_errs count_in_ref count_in_hyp",
        file=f,
    )
    for _, word, counts in sorted(
        ((sum(v[1:]), k, v) for k, v in words.items()), reverse=True
    ):
        corr, ref_sub, hyp_sub, ins_c, dels_c = counts
        word_tot_errs = ref_sub + hyp_sub + ins_c + dels_c
        ref_count = corr + ref_sub + dels_c
        hyp_count = corr + hyp_sub + ins_c
        print(f"{word}   {corr} {word_tot_errs} {ref_count} {hyp_count}", file=f)

    return float(tot_err_rate)
