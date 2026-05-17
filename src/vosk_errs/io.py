from pathlib import Path
from typing import Dict, List, Set, Tuple


def read_frequent_words(path: Path, top_n: int, lowercase: bool = False) -> Set[str]:
    """Read a `<logprob>\\t<word>` frequency file and return the top `top_n` words.

    The file is sorted by logprob descending (most frequent first) before
    truncation, so it doesn't need to be pre-sorted.
    """
    entries: List[Tuple[float, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    f"{path}:{lineno}: expected '<logprob> <word>', got {line!r}"
                )
            logprob = float(parts[0])
            word = parts[1]
            if lowercase:
                word = word.lower()
            entries.append((logprob, word))
    entries.sort(key=lambda e: e[0], reverse=True)
    return {w for _, w in entries[:top_n]}


def read_transcripts(path: Path) -> Dict[str, List[str]]:
    """Read a Kaldi-style transcript file: `<utt_id> w1 w2 ...` per line.

    Blank lines and lines starting with '#' are ignored. Duplicate utterance
    ids raise ValueError. An utterance with no words yields an empty list.
    """
    out: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            utt_id, words = parts[0], parts[1:]
            if utt_id in out:
                raise ValueError(
                    f"{path}:{lineno}: duplicate utterance id {utt_id!r}"
                )
            out[utt_id] = words
    return out


def pair(
    ref: Dict[str, List[str]],
    hyp: Dict[str, List[str]],
) -> Tuple[List[Tuple[str, List[str], List[str]]], List[str], List[str]]:
    """Join ref and hyp by utterance id.

    Returns (pairs, missing_in_hyp, extra_in_hyp). Pairs preserve ref order.
    Utterances present only in hyp are reported but not scored.
    """
    pairs = []
    missing = []
    for utt_id, ref_words in ref.items():
        if utt_id in hyp:
            pairs.append((utt_id, ref_words, hyp[utt_id]))
        else:
            missing.append(utt_id)
    extra = [u for u in hyp if u not in ref]
    return pairs, missing, extra
