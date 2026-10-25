"""Shared aggregation helpers used by the main study."""
from collections import Counter
import math


def entropy(answers):
    counts = Counter(answers)
    return -sum((n / len(answers)) * math.log2(n / len(answers)) for n in counts.values()) if answers else 0.0


def vote(answers):
    # Invalid outputs and abstentions are never votes for an option.
    counts = Counter(a for a in answers if a not in ('INVALID', 'ABSTAIN'))
    if not counts:
        return 'ABSTAIN'
    top = counts.most_common()
    return 'ABSTAIN' if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
