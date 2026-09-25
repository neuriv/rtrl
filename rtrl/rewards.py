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


def gsm8k_answer(text):
    """Last numeric answer, with explicit boxed answers taking precedence.

    Based on lm-evaluation-harness gsm8k.yaml flexible-extract (last match),
    extended with boxed priority, decimal equality and fraction rejection:
    https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gsm8k/gsm8k.yaml
    """
    import re
    from decimal import Decimal

    text = text.replace("−", "-").replace(r"\$", "").replace("$", "")
    text = re.sub(r"-\s+(?=\d)", "-", text)
    boxed = text.rfind(r"\boxed")
    if boxed >= 0:
        match = re.match(r"\\boxed\s*\{([^{}]*)\}", text[boxed:])
        if match is None:
            return None
        value = match[1].strip()
    else:
        matches = list(re.finditer(r"[-+]?(?:\d[\d,.]*|\.\d+)", text))
        if not matches:
            return None
        match = matches[-1]
        before, after = text[:match.start()], text[match.end():]
        # Do not treat the denominator of 1/2 or \frac{1}{2} as an answer.
        if (re.search(r"/\s*\(?\s*$|\\(?:[dt]?frac)\s*\{[^{}]*\}\s*\{\s*$", before)
                or re.match(r"\s*\)?\s*/", after)):
            return None
        value = match[0].rstrip(".,")
    if not re.fullmatch(r"[-+]?(?:(?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)", value):
        return None
    return Decimal(value.replace(",", ""))


def gsm8k(prompt, text):
    from decimal import Decimal
    answer = gsm8k_answer(text)
    return float(answer is not None and answer == Decimal(str(prompt["reference"]).replace(",", "")))
