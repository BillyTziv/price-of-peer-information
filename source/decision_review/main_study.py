"""Frozen main study. A failed shard is retained; never rerun a paid shard blindly."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random

from . import scenario_tasks_v3 as v3
from .experiment import entropy, vote
from .paired import METHODS, MockProvider, PairedRunner, digest, public_request

PROTOCOL = 'mba-economic-main-v1'
SEED = 2026101301
ORDER_SEED = 2026101302
BOOTSTRAP_SEED = 2026101303
DATASET_SHA256 = 'f9138a155c8f9ee173f163c69c4ee933eb1b88bbd50adb9309a9b101147963e3'
SHARDS = 24
TASKS_PER_SHARD = 8
REPEATS = 3
MAX_OUTPUT = 8192
TIMEOUT = 180
BUDGET_PER_SHARD_EUR = 3.0
STAGES = (('small', 0), ('strong', 0), *(('pool', i) for i in range(3)),
          ('self_correct', 0), *(('independent_revision', i) for i in range(3)),
          *(('debate', i) for i in range(3)))


def dataset():
    tasks = v3.generate(SEED, 32)
    if {t['question'] for t in tasks} & {t['question'] for t in v3.generate(v3.SCREEN_SEED, 4) + v3.generate(v3.CONFIRMATION_SEED, 8)}:
        raise ValueError('Main/development overlap')
    for task in tasks:
        task['split'] = 'main'
    if len(tasks) != 192 or digest(tasks) != DATASET_SHA256:
        raise ValueError('Main dataset changed')
    return tasks


def jobs(shard):
    if shard not in range(SHARDS):
        raise ValueError('Shard must be 0..23')
    tasks = dataset()[shard::SHARDS]
    if len(tasks) != TASKS_PER_SHARD:
        raise ValueError('Incorrect shard size')
    result = [(t, r) for t in tasks for r in range(REPEATS)]
    random.Random(digest([ORDER_SEED, shard])).shuffle(result)
    return result


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def run(shard, out, mode, fx=None, fx_date=None, provider_override=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    scheduled = jobs(shard)
    if provider_override is not None:
        provider = provider_override
    elif mode == 'openai':
        from .live_provider import OpenAIProvider
        provider = OpenAIProvider(BUDGET_PER_SHARD_EUR, fx, fx_date, MAX_OUTPUT, 1.0,
                                  timeout_seconds=TIMEOUT)
    elif mode == 'mock':
        provider = MockProvider()
    else:
        raise ValueError('Unknown mode')
    runner = PairedRunner(provider, out, ORDER_SEED)
    manifest = dict(protocol=PROTOCOL, status='running', mode=mode, shard=shard,
                    dataset_sha256=DATASET_SHA256, planned_blocks=len(scheduled),
                    completed_blocks=0, repeats=REPEATS, order=[(t['id'], r) for t, r in scheduled],
                    runner_sha256=hashlib.sha256(Path(__file__).with_name('paired.py').read_bytes()).hexdigest(),
                    adapter_sha256=hashlib.sha256(Path(__file__).with_name('live_provider.py').read_bytes()).hexdigest(),
                    provider_metadata=provider.metadata() if hasattr(provider, 'metadata') else None)
    (out / 'dataset.json').write_text(json.dumps(dataset(), ensure_ascii=False, indent=2), encoding='utf-8')
    try:
        for task, repeat in scheduled:
            runner.block(task, repeat)
            manifest['completed_blocks'] += 1
            (out / 'checkpoint.json').write_text(json.dumps({'completed_blocks': manifest['completed_blocks'],
                'last_task_id': task['id'], 'last_repeat': repeat}), encoding='utf-8')
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='stopped', error_type=type(exc).__name__,
                        note='Inspect raw calls before any continuation; never rerun this paid shard unchanged.')
        raise
    finally:
        manifest.update(provider_metadata=provider.metadata() if hasattr(provider, 'metadata') else None,
                        physical_calls=len(runner.calls), completed_episodes=len(runner.rows),
                        known_physical_cost_eur=provider.spent if hasattr(provider, 'spent') else sum(c['cost_eur'] for c in runner.calls.values()),
                        attributed_cost_eur=sum(r['cost_eur'] for r in runner.rows))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def audit_shard(path, expected_mode=None):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    shard = manifest['shard']
    scheduled = jobs(shard)
    calls = read_jsonl(path / 'calls.jsonl')
    rows = read_jsonl(path / 'episodes.jsonl')
    ids = {f'{task["id"]}/{repeat}/{stage}/{index}' for task, repeat in scheduled for stage, index in STAGES}
    byid = {c['call_id']: c for c in calls}
    keyed = {(r['task_id'], r['repeat'], r['method']): r for r in rows}
    if (manifest['protocol'] != PROTOCOL or manifest['status'] != 'complete'
        or manifest['dataset_sha256'] != DATASET_SHA256 or manifest['order'] != [[t['id'], r] for t, r in scheduled]
        or manifest['mode'] not in ('mock', 'openai') or (expected_mode and manifest['mode'] != expected_mode)
        or manifest['planned_blocks'] != 24 or manifest['completed_blocks'] != 24
        or manifest['physical_calls'] != 288 or manifest['completed_episodes'] != 144
        or len(calls) != 288 or len(byid) != 288 or set(byid) != ids
        or len(rows) != 144 or len(keyed) != 144
        or set(keyed) != {(t['id'], r, m) for t, r in scheduled for m in METHODS}
        or manifest['runner_sha256'] != hashlib.sha256(Path(__file__).with_name('paired.py').read_bytes()).hexdigest()
        or manifest['adapter_sha256'] != hashlib.sha256(Path(__file__).with_name('live_provider.py').read_bytes()).hexdigest()
        or json.loads((path / 'dataset.json').read_text()) != dataset()):
        raise ValueError(f'Incomplete or altered shard {shard}')
    meta = manifest.get('provider_metadata')
    if manifest['mode'] == 'openai':
        from .live_provider import MODELS, RATES
        from .scenario_confirmation import _live_call_ok
        recovery = manifest.get('transport_recovery')
        attempted = 289 if recovery else 288
        unresolved = recovery['unknown_reserved_eur'] if recovery else 0
        if (not meta or meta['models'] != MODELS or meta['rates_usd_per_million'] != {k: list(v) for k, v in RATES.items()}
            or meta['max_output_tokens'] != MAX_OUTPUT or meta['timeout_seconds'] != TIMEOUT
            or meta['retries'] != 0 or meta['dispatched_attempts'] != attempted
            or not math.isclose(meta['unknown_reserved_eur'], unresolved, rel_tol=1e-10, abs_tol=1e-12)
            or meta['halted'] or meta['temperature'] != 1):
            raise ValueError('Live provider controls or charges differ')
        if recovery:
            failed = read_jsonl(path / 'failed_attempts.jsonl')
            original = read_jsonl(path / 'original_calls.jsonl')
            original_rows = read_jsonl(path / 'original_episodes.jsonl')
            original_manifest = json.loads((path / 'original_manifest.json').read_text())
            if (len(failed) != 1 or len(original) != recovery['original_successful_calls']+1
                or len(original_rows) != recovery['original_episodes']
                or original_manifest['status'] != 'stopped' or original_manifest['shard'] != shard
                or original_manifest['physical_calls'] != recovery['original_successful_calls']
                or original_manifest['completed_episodes'] != recovery['original_episodes']
                or original_manifest['provider_metadata']['dispatched_attempts'] != recovery['original_successful_calls']+1
                or original[-1] != failed[0] or failed[0]['call_id'] != recovery['failed_call_id']
                or not failed[0].get('dispatched') or not failed[0].get('cost_unknown')
                or 'raw_response' in failed[0]
                or not math.isclose(failed[0]['reserved_eur'], unresolved, rel_tol=1e-10, abs_tol=1e-12)
                or any(c != byid.get(c['call_id']) for c in original[:-1])
                or any(r != keyed.get((r['task_id'], r['repeat'], r['method'])) for r in original_rows)):
                raise ValueError('Recovered shard did not preserve original attempts and episodes')
    for task, repeat in scheduled:
        def call(stage, index):
            return byid[f'{task["id"]}/{repeat}/{stage}/{index}']
        for stage, index in STAGES:
            rec = call(stage, index)
            source = 'small' if stage == 'self_correct' else 'pool'
            own = {k: call(source, index)[k] for k in ('rationale', 'answer')} if stage in ('self_correct', 'independent_revision', 'debate') else None
            peers = [{k: call('pool', j)[k] for k in ('rationale', 'answer')} for j in range(3) if j != index] if stage == 'debate' else []
            actual_peers = rec['request']['revision']['peers'] if stage == 'debate' else []
            if (Counter(map(digest, peers)) != Counter(map(digest, actual_peers))
                or rec['request'] != public_request(task, own, actual_peers)
                or rec['request_sha256'] != digest(rec['request'])
                or rec['requested_model'] != ('strong' if stage == 'strong' else 'small')
                or rec['answer'] not in (*task['options'], 'ABSTAIN', 'INVALID')
                or rec.get('error_type') or not isinstance(rec.get('rationale'), str)
                or not math.isfinite(rec['cost_eur']) or rec['cost_eur'] < 0
                or not math.isfinite(rec['latency_seconds']) or rec['latency_seconds'] < 0):
                raise ValueError(f'Bad call {rec["call_id"]}')
            if manifest['mode'] == 'openai':
                model = 'strong' if stage == 'strong' else 'small'
                if not _live_call_ok(rec, model, MAX_OUTPUT):
                    raise ValueError(f'Raw API mismatch {rec["call_id"]}')
                u = rec['usage']; cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
                a, b, c = meta['rates_usd_per_million'][model]
                charge = ((u['prompt_tokens']-cached)*a+cached*b+u['completion_tokens']*c)/1e6*meta['eur_per_usd']
                if not math.isclose(charge, rec['cost_eur'], rel_tol=1e-10, abs_tol=1e-12):
                    raise ValueError('Wrong charge')
        for method in METHODS:
            first = [call('strong' if method == 'single_strong' else 'small', 0)] if method in ('single_small','single_strong','self_correct') else [call('pool', i) for i in range(3)]
            last = first if method in ('single_small','single_strong','vote') else [call(method, i) for i in range(len(first))]
            path = first if first is last else first + last
            row = keyed[task['id'], repeat, method]
            initial = [c['answer'] for c in first]; final = [c['answer'] for c in last]
            a0 = vote(initial) if len(first)>1 else initial[0]
            a1 = vote(final) if len(last)>1 else final[0]
            expected = dict(initial_call_ids=[c['call_id'] for c in first], final_call_ids=[c['call_id'] for c in last],
                path_call_ids=[c['call_id'] for c in path], final_answer=a1, correct=a1 == task['answer'],
                answered=a1 in task['options'], initial_group_answer=a0, calls=len(path),
                entropy_initial=entropy(initial) if len(first)>1 else None,
                entropy_final=entropy(final) if len(last)>1 else None,
                unanimous_wrong=(len(set(final)) == 1 and final[0] in task['options'] and final[0] != task['answer']) if len(final)>1 else None,
                correct_to_incorrect=sum(a == task['answer'] and b != task['answer'] for a,b in zip(initial,final)) if first is not last else 0,
                incorrect_to_correct=sum(a != task['answer'] and b == task['answer'] for a,b in zip(initial,final)) if first is not last else 0,
                eligible_correct=sum(a == task['answer'] for a in initial) if first is not last else 0,
                eligible_incorrect=sum(a != task['answer'] for a in initial) if first is not last else 0,
                group_correct_to_incorrect=first is not last and a0 == task['answer'] and a1 != task['answer'],
                group_incorrect_to_correct=first is not last and a0 != task['answer'] and a1 == task['answer'])
            if (any(row.get(k) != v for k,v in expected.items())
                or not math.isclose(row['cost_eur'], sum(c['cost_eur'] for c in path), rel_tol=1e-10, abs_tol=1e-12)
                or not math.isclose(row['latency_seconds'], sum(c['latency_seconds'] for c in path), rel_tol=1e-10, abs_tol=1e-12)):
                raise ValueError(f'Bad episode {task["id"]}/{repeat}/{method}')
    if not math.isclose(sum(c['cost_eur'] for c in calls), manifest['known_physical_cost_eur'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Physical cost mismatch')
    if not math.isclose(sum(r['cost_eur'] for r in rows), manifest['attributed_cost_eur'], rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError('Attributed cost mismatch')
    return manifest, rows


def analyze(paths, mode='openai', bootstrap=10000):
    if len(paths) != SHARDS:
        raise ValueError('All 24 unique shards required')
    manifests, rows = [], []
    for path in paths:
        m, r = audit_shard(path, mode)
        manifests.append(m); rows.extend(r)
    if set(m['shard'] for m in manifests) != set(range(SHARDS)):
        raise ValueError('Duplicate/missing shard')
    invalid_calls = sum(c['answer'] == 'INVALID' for path in paths for c in read_jsonl(Path(path) / 'calls.jsonl'))
    truncated_calls = sum(c.get('finish_reason') == 'length' for path in paths
                          for c in read_jsonl(Path(path) / 'calls.jsonl'))
    sample = defaultdict(dict)
    for row in rows:
        sample[row['task_id'], row['repeat']][row['method']] = row
    if len(sample) != 576 or any(set(v) != set(METHODS) for v in sample.values()):
        raise ValueError('Missing complete paired blocks')
    summary = []
    for method in METHODS:
        arm = [r[method] for r in sample.values()]
        ncorrect = sum(r['correct'] for r in arm)
        cost = sum(r['cost_eur'] for r in arm)
        summary.append(dict(method=method, correct=ncorrect, episodes=len(arm), accuracy=ncorrect/len(arm),
            answered=sum(r['answered'] for r in arm), attributed_cost_eur=cost,
            coverage=sum(r['answered'] for r in arm)/len(arm),
            accuracy_when_answered=ncorrect/sum(r['answered'] for r in arm) if sum(r['answered'] for r in arm) else None,
            cost_per_correct_eur=cost/ncorrect if ncorrect else None,
            mean_serial_latency_seconds=sum(r['latency_seconds'] for r in arm)/len(arm),
            unanimous_wrong=sum(bool(r['unanimous_wrong']) for r in arm) if method in ('vote','independent_revision','debate') else None,
            group_harms=sum(bool(r['group_correct_to_incorrect']) for r in arm),
            group_repairs=sum(bool(r['group_incorrect_to_correct']) for r in arm),
            agent_correct_to_incorrect=sum(r['correct_to_incorrect'] for r in arm),
            agent_incorrect_to_correct=sum(r['incorrect_to_correct'] for r in arm)))
    wins = Counter()
    for pair in sample.values():
        d, ir = pair['debate']['correct'], pair['independent_revision']['correct']
        wins['both_correct' if d and ir else 'debate_only' if d else 'ir_only' if ir else 'both_noncorrect'] += 1
    taskdiff = defaultdict(list)
    for (tid, repeat), pair in sample.items():
        taskdiff[tid].append(int(pair['debate']['correct'])-int(pair['independent_revision']['correct']))
    families = defaultdict(list)
    for task in dataset():
        families[task['family']].append(sum(taskdiff[task['id']])/REPEATS)
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = []
    for _ in range(bootstrap):
        sampled = [rng.choice(values) for values in families.values() for __ in range(len(values))]
        deltas.append(sum(sampled)/len(sampled))
    deltas.sort()
    def quantile(q):
        pos = (len(deltas)-1)*q; i = int(pos)
        return deltas[i] + (deltas[min(i+1,len(deltas)-1)]-deltas[i])*(pos-i)
    by_template = defaultdict(list)
    task_by_id = {t['id']: t for t in dataset()}
    for tid, diffs in taskdiff.items():
        by_template[task_by_id[tid]['template']].append(sum(diffs)/REPEATS)
    template_effects = {template: dict(tasks=len(values), debate_minus_ir_accuracy=sum(values)/len(values))
                        for template,values in sorted(by_template.items())}
    qualified = [r for r in summary if r['accuracy'] >= .9]
    cheapest_qualified = min(qualified, key=lambda r:r['cost_per_correct_eur'])['method'] if qualified else None
    unknown = sum((m.get('transport_recovery') or {}).get('unknown_reserved_eur', 0) for m in manifests)
    by_method_unknown = Counter()
    for m in manifests:
        recovery = m.get('transport_recovery')
        if recovery:
            by_method_unknown[recovery['method']] += recovery['unknown_reserved_eur']
    for row in summary:
        row['uncertain_attempt_upper_bound_eur'] = by_method_unknown[row['method']]
        row['cost_per_correct_upper_bound_eur'] = ((row['attributed_cost_eur'] + by_method_unknown[row['method']])
            / row['correct'] if row['correct'] else None)
    return dict(protocol=PROTOCOL, mode=mode, dataset_sha256=DATASET_SHA256,
        unique_tasks=192, repeats=REPEATS, paired_blocks=len(sample), methods=summary,
        paired_counts=dict(wins), debate_minus_ir_accuracy=(wins['debate_only']-wins['ir_only'])/576,
        paired_cluster_bootstrap_95pct=[quantile(.025),quantile(.975)], bootstrap_draws=bootstrap,
        bootstrap_unit='task (three repetitions retained together; stratified by family)',
        sensitivity_by_template=template_effects,
        physical_cost_eur=sum(m['known_physical_cost_eur'] for m in manifests),
        attributed_cost_eur=sum(m['attributed_cost_eur'] for m in manifests),
        uncertain_transport_attempts=sum(bool(m.get('transport_recovery')) for m in manifests),
        unknown_transport_cost_upper_bound_eur=unknown,
        physical_cost_upper_bound_eur=sum(m['known_physical_cost_eur'] for m in manifests)+unknown,
        invalid_calls=invalid_calls, truncated_calls=truncated_calls,
        empirical_target_90pct=[r['method'] for r in qualified],
        cheapest_empirical_90pct_method=cheapest_qualified,
        note='A confidence interval crossing zero supports no directional superiority claim. No target is guaranteed by an empirical threshold alone.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard', type=int)
    p.add_argument('--out', type=Path)
    p.add_argument('--mode', choices=('mock','openai'), default='mock')
    p.add_argument('--fx', type=float)
    p.add_argument('--fx-date')
    p.add_argument('--analyze', nargs='*', type=Path)
    a = p.parse_args()
    if a.analyze is not None:
        print(json.dumps(analyze(a.analyze, a.mode), indent=2))
    elif a.shard is not None and a.out:
        print(json.dumps(run(a.shard, a.out, a.mode, a.fx, a.fx_date), indent=2))
    else:
        p.error('Provide --shard and --out, or --analyze with 24 paths')


if __name__ == '__main__':
    main()
