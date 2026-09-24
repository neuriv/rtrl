import pytest

from prepare_task import final_answer, prepare_rows
from rewards import arithmetic


@pytest.mark.parametrize("answer,expected", [("steps\n#### 1,024", "1024"),
                                            ("steps\n#### -30", "-30")])
def test_reference_matches_arithmetic(answer, expected):
    reference = final_answer(answer)
    assert reference == expected
    assert arithmetic({"reference": reference}, f"Reasoning.\n{expected}") == 1


@pytest.mark.parametrize("answer", ["42", "#### 1.5", "#### 1/2", "#### 1,23",
                                   "#### 42 extra", "#### __import__('os')"])
def test_reject_malformed_reference(answer):
    with pytest.raises(ValueError, match="Invalid GSM8K"):
        final_answer(answer)


def test_deterministic_disjoint_splits_and_original_indices():
    train = [{"question": "Train question?", "answer": "steps\n#### 3"}]
    test = [{"question": f"Test question {i}?", "answer": f"steps\n#### {i}"}
            for i in range(20)]
    first = prepare_rows(train, test, 5, 2718)
    assert first == prepare_rows(train, test, 5, 2718)
    assert first[2] != prepare_rows(train, test, 5, 2719)[2]
    assert len(first[0]) == 1 and len(first[1]) == 5
    assert {row["id"] for row in first[0]}.isdisjoint(row["id"] for row in first[1])
    assert [row["reference"] for row in first[1]] == [str(i) for i in first[2]["eval"]]
    with pytest.raises(ValueError, match="overlap"):
        prepare_rows(train, train, 1, 2718)
    with pytest.raises(ValueError, match="eval-size"):
        prepare_rows(train, test, 21, 2718)
