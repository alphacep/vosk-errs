from io import StringIO

from vosk_errs.report import write_error_stats


def test_rare_word_substitutions_are_reported_when_frequency_list_is_used():
    out = StringIO()
    write_error_stats(
        out,
        [
            ("utt1", ["common", "rare", "rare"], ["common", "wrong", "wrong"]),
            ("utt2", ["common"], ["wrong"]),
        ],
        frequent_words={"common"},
        top_n=1,
    )

    report = out.getvalue()
    assert "SUBSTITUTIONS: count ref -> hyp\n2   rare -> wrong\n1   common -> wrong" in report
    assert "RARE WORD SUBSTITUTIONS: count ref -> hyp\n2   rare -> wrong" in report
    assert "common -> wrong" not in report.split("RARE WORD SUBSTITUTIONS:", 1)[1].split(
        "DELETIONS:", 1
    )[0]


def test_rare_word_substitutions_are_omitted_without_frequency_list():
    out = StringIO()
    write_error_stats(
        out,
        [("utt1", ["rare"], ["wrong"])],
    )

    assert "RARE WORD SUBSTITUTIONS" not in out.getvalue()


def test_rare_word_substitutions_use_explicit_rare_word_list():
    out = StringIO()
    write_error_stats(
        out,
        [
            ("utt1", ["common", "rare", "rare"], ["wrong", "wrong", "wrong"]),
            ("utt2", ["other"], ["wrong"]),
        ],
        rare_words={"rare"},
    )

    report = out.getvalue()
    assert "SRATE: 100.00 (4/4) 100.00 (2/2)" in report
    assert "RARE WORD SUBSTITUTIONS: count ref -> hyp\n2   rare -> wrong" in report
    rare_section = report.split("RARE WORD SUBSTITUTIONS:", 1)[1].split(
        "DELETIONS:", 1
    )[0]
    assert "common -> wrong" not in rare_section
    assert "other -> wrong" not in rare_section
