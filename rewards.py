"""Minimal verifier for exact-answer tasks; custom verifiers use the same signature."""


def exact_match(prompt, text):
    """Requires a reference field; does not claim mathematical equivalence."""
    return float(" ".join(text.split()) == " ".join(str(prompt["reference"]).split()))


def final_integer(prompt, text):
    """Task contract: terminate with `Final: <integer>` on its own line."""
    import re
    match = re.search(r"(?:^|\n)Final: (-?\d+)\s*\Z", text)
    return float(match is not None and int(match[1]) == int(prompt["reference"]))


def terminal_integer(text):
    """A terminal integer, allowing thousands separators and trailing markup."""
    import re
    text = re.sub(r"-\s+(?=\d)", "-", text.replace("−", "-"))
    text = re.sub(r"\\boxed\{(-?[\d,]+)\}", r"\1", text)
    match = re.search(r"(?:^|[^\w.,/\-^{])(-?(?:\d{1,3}(?:,\d{3})+|\d+))[\s.*$\\\]\)}]*\Z", text)
    return int(match[1].replace(",", "")) if match else None


def arithmetic(prompt, text):
    return float(terminal_integer(text) == int(prompt["reference"]))
