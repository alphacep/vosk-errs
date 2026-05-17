from vosk_errs.align import Op, align, score


def test_perfect_match():
    s, a = score(["a", "b", "c"], ["a", "b", "c"])
    assert s.hits == 3 and s.errors == 0 and s.wer == 0.0
    assert all(op is Op.EQUAL for op, _, _ in a)


def test_substitution():
    s, _ = score(["the", "cat", "sat"], ["the", "bat", "sat"])
    assert (s.hits, s.subs, s.inss, s.dels) == (2, 1, 0, 0)
    assert abs(s.wer - 1 / 3) < 1e-9


def test_insertion():
    s, _ = score(["a", "b"], ["a", "x", "b"])
    assert (s.hits, s.subs, s.inss, s.dels) == (2, 0, 1, 0)


def test_deletion():
    s, _ = score(["a", "b", "c"], ["a", "c"])
    assert (s.hits, s.subs, s.inss, s.dels) == (2, 0, 0, 1)


def test_empty_ref():
    s, _ = score([], ["a", "b"])
    assert s.inss == 2 and s.ref_words == 0


def test_empty_hyp():
    s, _ = score(["a", "b"], [])
    assert s.dels == 2 and s.wer == 1.0


def test_alignment_shape():
    _, a = score(["the", "quick", "brown", "fox"],
                 ["the", "quik", "fox", "jumps"])
    # one sub, one del, one ins -> 4 + adjustments
    ops = [op for op, _, _ in a]
    assert ops.count(Op.EQUAL) == 2
    assert ops.count(Op.SUB) == 1
