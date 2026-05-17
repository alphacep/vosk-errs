from dataclasses import dataclass, field
from enum import Enum
from typing import List, Sequence, Tuple, Optional


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

    Substitutions, insertions and deletions all cost 1.
    """
    n, m = len(ref), len(hyp)
    # dp[i][j] = edit distance between ref[:i] and hyp[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i][0] = i
    for j in range(1, m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j - 1],  # sub
                    dp[i - 1][j],      # del
                    dp[i][j - 1],      # ins
                )

    out: Alignment = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            out.append((Op.EQUAL, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            out.append((Op.SUB, ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            out.append((Op.DEL, ref[i - 1], None))
            i -= 1
        else:
            out.append((Op.INS, None, hyp[j - 1]))
            j -= 1
    out.reverse()
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
