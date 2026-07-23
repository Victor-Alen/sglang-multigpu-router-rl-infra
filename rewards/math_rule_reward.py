from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_ANSWER = re.compile(r"(?:answer\s*(?:is|:)|####)\s*([^\n]+)", re.IGNORECASE)


def extract_answer(text: str) -> Optional[str]:
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    answers = _ANSWER.findall(text)
    if answers:
        return answers[-1].strip().rstrip(".")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1].rstrip(".") if lines else None


def _number(value: str) -> Optional[Decimal]:
    try:
        return Decimal(value.replace(",", "").strip())
    except (InvalidOperation, AttributeError):
        return None


def math_rule_reward(response: str, reference: str) -> float:
    predicted = extract_answer(response)
    expected = extract_answer(reference) or reference.strip()
    if predicted is None:
        return 0.0
    left, right = _number(predicted), _number(expected)
    if left is not None and right is not None:
        return 1.0 if left == right else 0.0
    return 1.0 if predicted.casefold() == expected.casefold() else 0.0
