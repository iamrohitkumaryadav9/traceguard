import pytest

from src.rag.matching import contains_phrase, matches_all, normalize


def test_case_and_whitespace_are_normalised():
    assert contains_phrase("The   Funnel\n Account  activity", "funnel account")
    assert normalize("  A\tB \n C ") == "a b c"


def test_hyphen_slash_space_are_equivalent():
    assert contains_phrase("5.6 Prohibition on Tipping-off", "tipping off")
    assert contains_phrase("the fact of tipping off the client", "tipping-off")
    assert contains_phrase("7.5 Approach on Tumbler/Mixer and other", "tumbler mixer")
    assert contains_phrase("tumbler mixer", "Tumbler/Mixer")
    assert contains_phrase("ransomware-related cases", "ransomware related cases")
    assert contains_phrase("see CVC FIN-2019-A003 here", "CVC FIN 2019 A003")


def test_soft_hyphens_are_ignored():
    assert contains_phrase("the tip­ping off rule", "tipping off")
    assert contains_phrase("the tipping off rule", "tip­ping off")


def test_hyphenation_at_line_breaks():
    assert contains_phrase("this is ransom-\nware activity", "ransomware")
    assert contains_phrase("this is ransom- ware activity", "ransomware")  # lines already joined with a space
    assert contains_phrase("anti-\nmoney laundering", "anti money laundering")  # a real hyphen still matches


def test_alternatives_and_all_phrases():
    assert contains_phrase("incoming from disparate sources", "accumulation of funds|disparate sources")
    assert not contains_phrase("nothing relevant", "accumulation of funds|disparate sources")
    assert matches_all("a red flag and a mixer", ["red flag", "mixer"])
    assert not matches_all("a red flag", ["red flag", "mixer"])


def test_no_false_positives_on_partial_words():
    assert not contains_phrase("tipping offer", "tipping off")  # 'tipping offer' contains 'tipping off' as a prefix
    assert contains_phrase("rapid trades between coins", "rapid trades")
    assert not contains_phrase("rapid trade", "rapid trades")
    with pytest.raises(AttributeError):
        contains_phrase(None, "x")
