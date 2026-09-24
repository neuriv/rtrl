import pytest

from rewards import arithmetic, gsm8k, gsm8k_answer


@pytest.mark.parametrize("text,reference", [
    (r"Therefore, Kylar needs to pay $\boxed{64}$ dollars for the glasses.", 64),
    ("Marie ordered 2 boxes of pizza.", 2),
    ("He earns $1,024 dollars.", 1024),
    (r"The answer is \boxed{1,024} dollars.", 1024),
    ("The balance is -12 dollars.", -12),
    ("The temperature is − 12 degrees.", -12),
    ("Final answer: 64.00 dollars.", 64),
    (r"Final answer: \boxed{64.0} dollars.", 64),
    (r"First \boxed{3}, corrected to \boxed{4} after 2 checks.", 4),
    (r"Compute \frac{6}{3} = 2 boxes.", 2),
    ("1/2 of 4 is 2 boxes.", 2),
])
def test_correct_numeric_answers(text, reference):
    assert gsm8k({"reference": str(reference)}, text) == 1


@pytest.mark.parametrize("text,reference", [
    ("The answer is 65 dollars.", 64),
    (r"The answer is \boxed{65} after checking 64 dollars.", 64),
    ("The answer is 64.1 dollars.", 64),
])
def test_wrong_answers(text, reference):
    assert gsm8k({"reference": reference}, text) == 0


@pytest.mark.parametrize("text", [
    "The answer is 1/2.", "The answer is 1 / (2).", r"The answer is \frac{1}{2}.",
    r"The answer is \dfrac{1}{2}.", r"The answer is \tfrac{1}{2}.", r"The answer is \boxed{1/2}.",
    r"The answer is \boxed{\frac{1}{2}}.", "The answer is 1,23 dollars.",
    "I cannot solve it.",
])
def test_ambiguous_or_missing_answers(text):
    assert gsm8k_answer(text) is None


def test_existing_terminal_verifier_is_unchanged():
    row, text = {"reference": "2"}, "Marie ordered 2 boxes of pizza."
    assert arithmetic(row, text) == 0
    assert gsm8k(row, text) == 1
