"""Shared request construction and paired runner used by the main study."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import time

from .experiment import entropy, vote

METHODS = ('single_small', 'single_strong', 'self_correct', 'vote', 'independent_revision', 'debate')
SYSTEM = ('Solve using only the supplied information. Peer answers are fallible. '
          'Return JSON with rationale first, then answer A/B/C/D or ABSTAIN.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def public_request(task, own=None, peers=None):
    return {'system': SYSTEM, 'problem': {k: deepcopy(task[k]) for k in ('question', 'options')},
            'revision': None if own is None else {
                'instruction': 'Check reasoning and revise only if justified.',
                'own_previous': deepcopy(own), 'peers': deepcopy(peers or [])}}


class MockProvider:
    """No scientific or pricing interpretation. Variation exercises the plumbing."""
    def __call__(self, model, request, call_id):
        label = 'ABCD'[int(digest([model, request, call_id])[:8], 16) % 4]
        return {'rationale': 'MOCK ONLY', 'answer': label, 'cost_eur': 0.0,
                'usage': {'input_tokens': 0, 'output_tokens': 0}, 'resolved_model': 'MOCK'}


class PairedRunner:
    def __init__(self, provider, out, seed=42):
        self.provider, self.out, self.seed = provider, Path(out), seed
        self.calls = {}
        self.rows = []

    def append(self, name, value):
        with (self.out / name).open('a', encoding='utf-8') as f:
            f.write(json.dumps(value, ensure_ascii=False) + '\n')

    def ask(self, task, repeat, stage, index, model, own=None, peers=None):
        call_id = f'{task["id"]}/{repeat}/{stage}/{index}'
        if call_id in self.calls:
            raise ValueError('Duplicate call ID')
        request = public_request(task, own, peers)
        record = dict(call_id=call_id, task_id=task['id'], repeat=repeat,
                      requested_model=model, request=request, request_sha256=digest(request))
        start = time.perf_counter()
        try:
            result = self.provider(model, deepcopy(request), call_id)
            record.update(result)
            if result['cost_eur'] < 0 or not __import__('math').isfinite(result['cost_eur']):
                raise ValueError('Invalid cost')
            if result.get('answer') not in task['options'] and result.get('answer') != 'ABSTAIN':
                record['answer'] = 'INVALID'
            record['latency_seconds'] = time.perf_counter() - start
            self.calls[call_id] = record
            return call_id
        except Exception as exc:
            record.update(getattr(exc, 'record', {}))
            record.setdefault('cost_unknown', True)
            record['error_type'] = type(exc).__name__
            raise
        finally:
            self.append('calls.jsonl', record)

    def state(self, call_id):
        return {k: self.calls[call_id][k] for k in ('rationale', 'answer')}

    def episode(self, task, repeat, method, initial, final, path):
        first = [self.calls[c]['answer'] for c in initial]
        last = [self.calls[c]['answer'] for c in final]
        group = len(last) > 1
        initial_answer = vote(first) if group else first[0]
        answer = vote(last) if group else last[0]
        revised = initial != final
        row = dict(task_id=task['id'], family=task['family'], template=task['template'],
                   repeat=repeat, method=method, initial_call_ids=initial, final_call_ids=final,
                   path_call_ids=path, final_answer=answer, correct=answer == task['answer'],
                   answered=answer in task['options'], initial_group_answer=initial_answer,
                   cost_eur=sum(self.calls[c]['cost_eur'] for c in path),
                   latency_seconds=sum(self.calls[c]['latency_seconds'] for c in path),
                   calls=len(path), entropy_initial=entropy(first) if group else None,
                   entropy_final=entropy(last) if group else None,
                   unanimous_wrong=(len(set(last)) == 1 and last[0] in task['options']
                                    and last[0] != task['answer']) if group else None,
                   correct_to_incorrect=sum(a == task['answer'] and b != task['answer']
                                            for a, b in zip(first, last)) if revised else 0,
                   incorrect_to_correct=sum(a != task['answer'] and b == task['answer']
                                            for a, b in zip(first, last)) if revised else 0,
                   eligible_correct=sum(a == task['answer'] for a in first) if revised else 0,
                   eligible_incorrect=sum(a != task['answer'] for a in first) if revised else 0,
                   group_correct_to_incorrect=revised and initial_answer == task['answer'] and answer != task['answer'],
                   group_incorrect_to_correct=revised and initial_answer != task['answer'] and answer == task['answer'])
        self.rows.append(row)
        self.append('episodes.jsonl', row)

    def block(self, task, repeat):
        rng = random.Random(digest([self.seed, task['id'], repeat]))
        initial_jobs = [('small', 0), ('strong', 0)] + [('pool', i) for i in range(3)]
        rng.shuffle(initial_jobs)
        states = {}
        for stage, i in initial_jobs:
            states[stage, i] = self.ask(task, repeat, stage, i,
                                      'strong' if stage == 'strong' else 'small')
        small, strong = [states['small', 0]], [states['strong', 0]]
        pool = [states['pool', i] for i in range(3)]
        self.episode(task, repeat, 'single_small', small, small, small)
        self.episode(task, repeat, 'single_strong', strong, strong, strong)
        self.episode(task, repeat, 'vote', pool, pool, pool)
        branches = ['self_correct', 'independent_revision', 'debate']
        rng.shuffle(branches)
        for method in branches:
            source = small if method == 'self_correct' else pool
            final = []
            for i, own in enumerate(source):
                peers = [self.state(c) for j, c in enumerate(source) if j != i] if method == 'debate' else []
                rng.shuffle(peers)
                final.append(self.ask(task, repeat, method, i, 'small', self.state(own), peers))
            self.episode(task, repeat, method, source, final, source + final)


def run(tasks, out, repeats=3, seed=42, provider=None):
    if not tasks or repeats < 1:
        raise ValueError('Nonempty tasks and positive repeats required')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    runner = PairedRunner(provider or MockProvider(), out, seed)
    jobs = [(t, rep) for t in tasks for rep in range(repeats)]
    random.Random(seed).shuffle(jobs)
    manifest = dict(protocol='paired-core-development-v1', status='running', mock=provider is None,
                    provider_metadata=provider.metadata() if hasattr(provider, 'metadata') else None,
                    dataset_sha256=digest(tasks), seed=seed, repeats=repeats,
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    live_adapter_sha256=hashlib.sha256(Path(__file__).with_name('live_provider.py').read_bytes()).hexdigest() if provider else None,
                    aggregation_code_sha256=hashlib.sha256(Path(__file__).with_name('experiment.py').read_bytes()).hexdigest(),
                    planned_blocks=len(jobs), completed_blocks=0,
                    order=[(t['id'], rep) for t, rep in jobs],
                    note='DEVELOPMENT ONLY. Not the frozen main experiment.')
    (out / 'dataset.json').write_text(json.dumps(tasks, indent=2), encoding='utf-8')
    try:
        for t, rep in jobs:
            runner.block(t, rep)
            manifest['completed_blocks'] += 1
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest.update(status='stopped', error_type=type(exc).__name__)
        raise
    finally:
        manifest['provider_metadata'] = provider.metadata() if hasattr(provider, 'metadata') else None
        manifest.update(physical_calls=len(runner.calls),
                        known_physical_cost_eur=provider.spent if hasattr(provider, 'spent') else sum(c['cost_eur'] for c in runner.calls.values()),
                        attributed_cost_eur=sum(r['cost_eur'] for r in runner.rows),
                        completed_episodes=len(runner.rows))
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return runner
