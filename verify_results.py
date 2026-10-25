#!/usr/bin/env python3
"""Offline checks from retained dataset and derived decision records; no API calls."""
import ast
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'source'))
from decision_review.main_study import dataset, BOOTSTRAP_SEED
from decision_review.paired import digest

def check(condition, message):
    if not condition:
        raise ValueError(message)

def aggregate(labels):
    counts = Counter(x for x in labels if x in 'ABCD' and len(x) == 1)
    if not counts:
        return 'ABSTAIN'
    top = max(counts.values())
    leaders = [x for x, count in counts.items() if count == top]
    return leaders[0] if len(leaders) == 1 else 'ABSTAIN'

def main():
    tasks = json.loads((ROOT / 'evidence/main_dataset.json').read_text())
    check(tasks == dataset(), 'Generated tasks differ from retained dataset')
    by_id = {t['id']: t for t in tasks}
    rows = list(csv.DictReader((ROOT / 'evidence/paired_transitions.csv').open()))
    check(len(rows) == 576, 'Wrong number of paired blocks')
    seen = set()
    counts = {'IR3': Counter(), 'D3': Counter()}
    transitions = Counter()
    task_diffs = defaultdict(list)
    for row in rows:
        tid = row['task_id']
        check(tid in by_id, 'Unknown task')
        check(row['gold'] == by_id[tid]['answer'], 'Reference answer mismatch')
        key = (tid, int(row['repeat']))
        check(key not in seen, 'Duplicate paired block')
        seen.add(key)
        for prefix, method in [('ir', 'IR3'), ('d', 'D3')]:
            labels = ast.literal_eval(row[prefix + '_labels'])
            check(len(labels) == 3, 'Wrong agent count')
            final = aggregate(labels)
            check(final == row[prefix + '_final'], 'Aggregation mismatch')
            state = 'no_answer' if final == 'ABSTAIN' else 'correct' if final == row['gold'] else 'wrong'
            check(state == row[prefix + '_state'], 'State mismatch')
            counts[method][state] += 1
            if len(set(labels)) == 1 and labels[0] in ['A', 'B', 'C', 'D'] and labels[0] != row['gold']:
                counts[method]['unanimous_wrong'] += 1
        transitions[row['ir_state'] + ' -> ' + row['d_state']] += 1
        task_diffs[tid].append(int(row['d_state'] == 'correct') - int(row['ir_state'] == 'correct'))
    check(seen == {(t['id'], r) for t in tasks for r in range(3)}, 'Incomplete task repetitions')
    families = defaultdict(list)
    for task in tasks:
        families[task['family']].append(sum(task_diffs[task['id']]) / 3)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(10000):
        sample = [rng.choice(values) for values in families.values() for __ in range(len(values))]
        draws.append(sum(sample) / len(sample))
    draws.sort()
    def quantile(q):
        pos = (len(draws) - 1) * q
        i = int(pos)
        return draws[i] + (draws[min(i+1, len(draws)-1)] - draws[i]) * (pos-i)
    ci = [quantile(.025), quantile(.975)]
    delta = (counts['D3']['correct'] - counts['IR3']['correct']) / len(rows)
    reported = json.loads((ROOT / 'evidence/reproduced-main-analysis.json').read_text())
    for method, original in [('IR3', 'independent_revision'), ('D3', 'debate')]:
        expected = next(x for x in reported['methods'] if x['method'] == original)
        check(counts[method]['correct'] == expected['correct'], 'Correct count mismatch')
        check(counts[method]['unanimous_wrong'] == expected['unanimous_wrong'], 'Unanimous error mismatch')
        check(len(rows) - counts[method]['no_answer'] == expected['answered'], 'Coverage mismatch')
    check(math.isclose(delta, reported['debate_minus_ir_accuracy']), 'Primary difference mismatch')
    check(all(math.isclose(a, b) for a, b in zip(ci, reported['paired_cluster_bootstrap_95pct'])), 'Bootstrap interval mismatch')
    task_rows = list(csv.DictReader((ROOT / 'evidence/task_level_comparison.csv').open()))
    check(len(task_rows) == 192 and {r['task_id'] for r in task_rows} == set(by_id), 'SC6 task table mismatch')
    sc6 = json.loads((ROOT / 'evidence/six_sample_comparison.json').read_text())
    sc6_checks = {}
    for column in ['D3_minus_SC6_correct', 'D3_minus_SC6_uniform']:
        value = sum(float(r[column]) for r in task_rows) / len(task_rows)
        check(math.isclose(value, sc6['D3_minus_SC6'][column]['difference']), 'SC6 mean difference mismatch')
        sc6_checks[column] = value * 100
    result = dict(status='PASS', dataset_sha256=digest(tasks), tasks=len(tasks), paired_blocks=len(rows),
                  outcomes=counts, transitions=dict(sorted(transitions.items())),
                  accuracy_difference_percentage_points=delta*100,
                  task_cluster_95pct_percentage_points=[x*100 for x in ci],
                  sc6_mean_difference_percentage_points=sc6_checks,
                  scope='Checks retained task data, reference validator, derived final votes, primary bootstrap, and SC6 task-level mean differences. Does not re-audit raw API responses or token usage; does not reconstruct SC6 pools from raw outputs.')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
