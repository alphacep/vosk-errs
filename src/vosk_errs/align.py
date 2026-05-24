import difflib
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence, Tuple, Optional

from Levenshtein import opcodes as lev_opcodes


class Op(str, Enum):
    EQUAL = "="
    SUB = "S"
    INS = "I"
    DEL = "D"


@dataclass
class Stats:
    hits: int = 0
    subs: int = 0
    inss: int = 0
    dels: int = 0
    ref_words: int = 0

    def __iadd__(self, other: "Stats") -> "Stats":
        self.hits += other.hits
        self.subs += other.subs
        self.inss += other.inss
        self.dels += other.dels
        self.ref_words += other.ref_words
        return self

    @property
    def errors(self) -> int:
        return self.subs + self.inss + self.dels

    @property
    def wer(self) -> float:
        if self.ref_words == 0:
            return float("inf") if self.errors else 0.0
        return self.errors / self.ref_words


Alignment = List[Tuple[Op, Optional[str], Optional[str]]]


def align(ref: Sequence[str], hyp: Sequence[str]) -> Alignment:
    """Levenshtein alignment of `ref` vs `hyp` returning per-token ops.

    Substitutions, insertions and deletions all cost 1. Within Levenshtein
    `replace` blocks, an LCS sub-alignment is attempted to expose matches
    that the straight 1:1 mapping would obscure (kept only when it does
    not increase the op count).
    """
    ref = list(ref)
    hyp = list(hyp)
    out: Alignment = []
    for tag, si, se, di, de in lev_opcodes(ref, hyp):
        if tag == "equal":
            for k in range(se - si):
                out.append((Op.EQUAL, ref[si + k], hyp[di + k]))
        elif tag == "delete":
            for k in range(si, se):
                out.append((Op.DEL, ref[k], None))
        elif tag == "insert":
            for k in range(di, de):
                out.append((Op.INS, None, hyp[k]))
        else:  # 'replace' — Levenshtein guarantees se-si == de-di
            out.extend(_refine_replace(ref[si:se], hyp[di:de]))
    return out


def _refine_replace(ref_seg: List[str], hyp_seg: List[str]) -> Alignment:
    """Try difflib's LCS alignment on a replace block to expose matches.
    Falls back to straight SUBs if it would inflate the op count."""
    n = len(ref_seg)  # == len(hyp_seg)
    refined = _difflib_align(ref_seg, hyp_seg)
    if sum(1 for op, _, _ in refined if op is not Op.EQUAL) <= n:
        return refined
    return [(Op.SUB, ref_seg[k], hyp_seg[k]) for k in range(n)]


def _difflib_align(ref_seg: List[str], hyp_seg: List[str]) -> Alignment:
    out: Alignment = []
    sm = difflib.SequenceMatcher(a=ref_seg, b=hyp_seg, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        r, h = i2 - i1, j2 - j1
        if tag == "equal":
            for k in range(r):
                out.append((Op.EQUAL, ref_seg[i1 + k], hyp_seg[j1 + k]))
        elif tag == "replace":
            mn = min(r, h)
            for k in range(mn):
                out.append((Op.SUB, ref_seg[i1 + k], hyp_seg[j1 + k]))
            for k in range(mn, r):
                out.append((Op.DEL, ref_seg[i1 + k], None))
            for k in range(mn, h):
                out.append((Op.INS, None, hyp_seg[j1 + k]))
        elif tag == "delete":
            for k in range(r):
                out.append((Op.DEL, ref_seg[i1 + k], None))
        else:  # insert
            for k in range(h):
                out.append((Op.INS, None, hyp_seg[j1 + k]))
    return out


def score(ref: Sequence[str], hyp: Sequence[str]) -> Tuple[Stats, Alignment]:
    a = align(ref, hyp)
    s = Stats(ref_words=len(ref))
    for op, _, _ in a:
        if op is Op.EQUAL:
            s.hits += 1
        elif op is Op.SUB:
            s.subs += 1
        elif op is Op.INS:
            s.inss += 1
        elif op is Op.DEL:
            s.dels += 1
    return s, a
