"""Auditable v2 DEVELOPMENT scenarios. No model calls or main evaluation data."""
import argparse
from collections import Counter
from fractions import Fraction
import json
from pathlib import Path
import random


VERSION = 'economic-scenarios-v2.1'
DEFAULT_SEED = 2026101101


def profit(p, q):
    """Integer cents; all-unit price/cost changes are distinct from marginal overtime."""
    price = p['price'] - (p['price_cut'] if q >= p['price_threshold'] else 0)
    unit = p['unit'] - (p['unit_discount'] if q >= p['discount_threshold'] else 0)
    batches = (q + p['batch_size'] - 1) // p['batch_size']
    return (q * (price - unit) - p['fixed']
            - (p['setup'] if q > p['setup_threshold'] else 0)
            - p['overtime'] * max(q - p['overtime_threshold'], 0)
            - batches * p['batch_fee'])


def break_even(p):
    return next((q for q in range(p['capacity'] + 1) if profit(p, q) >= 0), None)


def break_even_piecewise(p):
    """Independent closed-form solver on linear intervals, not a call to profit()."""
    starts = {0, p['capacity'] + 1, p['price_threshold'], p['discount_threshold'],
              p['setup_threshold'] + 1, p['overtime_threshold'] + 1}
    starts.update(range(1, p['capacity'] + 1, p['batch_size']))
    starts = sorted(x for x in starts if 0 <= x <= p['capacity'] + 1)
    for lo, stop in zip(starts, starts[1:]):
        hi = stop - 1
        revenue_rate = p['price'] if lo < p['price_threshold'] else p['price'] - p['price_cut']
        purchase_rate = p['unit'] if lo < p['discount_threshold'] else p['unit'] - p['unit_discount']
        marginal_overtime = p['overtime'] if lo > p['overtime_threshold'] else 0
        slope = revenue_rate - purchase_rate - marginal_overtime
        intercept = (-p['fixed'] - (p['setup'] if lo > p['setup_threshold'] else 0)
                     - p['batch_fee'] * ((lo + p['batch_size'] - 1) // p['batch_size'])
                     + marginal_overtime * p['overtime_threshold'])
        if slope * lo + intercept >= 0:
            return lo
        if slope > 0:
            candidate = max(lo, -(intercept // slope))  # ceil(-intercept / slope)
            if candidate <= hi:
                return candidate
    return None


def supplier_scores(p):
    """Exact expected cents and chance constraint over supplied JOINT scenarios."""
    scores = []
    for plan in p['plans']:
        quantities = plan['quantities']
        capacity_ok = sum(quantities) == p['demand']
        base = 0
        for q, s in zip(quantities, p['suppliers']):
            capacity_ok &= q <= s['capacity']
            if q:
                rate = s['discounted_unit'] if q >= s['discount_threshold'] else s['unit']
                base += q * rate + s['fee']
        expected_shortage, service = Fraction(0), Fraction(0)
        for state in p['states']:
            delivered = sum(Fraction(q * pct, 100) for q, pct in zip(quantities, state['on_time_pct']))
            probability = Fraction(state['probability_pct'], 100)
            expected_shortage += probability * max(p['demand'] - delivered, 0)
            service += probability * (delivered >= p['service_units'])
        scores.append(dict(name=plan['name'], feasible=capacity_ok and service >= Fraction(p['minimum_service_pct'], 100),
                           base_cents=Fraction(base), service_probability=service,
                           expected_shortage=expected_shortage,
                           expected_cost_cents=base + p['emergency_unit'] * expected_shortage))
    return scores


def supplier_scores_enumerated(p):
    """Independent unit-by-unit and 100 equally likely tickets implementation.

    Development quantities are integral and on-time percentages produce integer units.
    This verifier neither calls supplier_scores nor reuses its intermediate outputs.
    """
    scores = []
    for plan in p['plans']:
        booked = 0
        for i, amount in enumerate(plan['quantities']):
            supplier = p['suppliers'][i]
            rate = supplier['unit']
            if amount >= supplier['discount_threshold']:
                rate = supplier['discounted_unit']
            booked += sum(rate for _ in range(amount))
            if amount > 0:
                booked += supplier['fee']
        tickets, fulfilled = [], 0
        for state in p['states']:
            deliveries = []
            for amount, pct in zip(plan['quantities'], state['on_time_pct']):
                if amount * pct % 100:
                    raise ValueError('Verifier requires integer delivered units')
                deliveries.extend([1] * (amount * pct // 100))
            units = len(deliveries)
            cost = booked + sum(p['emergency_unit'] for _ in range(max(p['demand'] - units, 0)))
            tickets.extend([cost] * state['probability_pct'])
            if units >= p['service_units']:
                fulfilled += state['probability_pct']
        if len(tickets) != 100:
            raise ValueError('Scenario probabilities must sum to 100%')
        feasible = (sum(plan['quantities']) == p['demand'] and
                    all(q <= s['capacity'] for q, s in zip(plan['quantities'], p['suppliers'])) and
                    fulfilled >= p['minimum_service_pct'])
        scores.append((plan['name'], feasible, Fraction(sum(tickets), 100), Fraction(fulfilled, 100)))
    return scores


def supplier_gold(p):
    eligible = sorted((s['expected_cost_cents'], s['name']) for s in supplier_scores(p) if s['feasible'])
    if len(eligible) < 2 or eligible[0][0] == eligible[1][0]:
        raise ValueError('Need at least two feasible plans and a unique best offered plan')
    return eligible[0][1]


def money(cents):
    return f'{cents // 100}.{cents % 100:02d}'


def break_even_question(p):
    return (f'All q units produced are sold, where q is an integer from 0 to {p["capacity"]}. '
            f'The selling price is EUR {money(p["price"])} per unit; when q >= {p["price_threshold"]}, '
            f'reduce that price by EUR {money(p["price_cut"])} on EVERY unit. '
            f'The material cost is EUR {money(p["unit"])} per unit; when q >= {p["discount_threshold"]}, '
            f'reduce that cost by EUR {money(p["unit_discount"])} on EVERY unit. '
            f'Fixed cost is EUR {money(p["fixed"])} even at q=0. '
            f'Add a one-time setup charge of EUR {money(p["setup"])} only when q > {p["setup_threshold"]}. '
            f'Each unit ABOVE {p["overtime_threshold"]} also costs EUR {money(p["overtime"])} in overtime. '
            f'Shipping costs EUR {money(p["batch_fee"])} per started batch of {p["batch_size"]} units '
            '(round the number of batches upward; zero units means zero batches). '
            'There are no other costs or taxes. What is the SMALLEST q at which total revenue is at least total cost? '
            'Profit may fall again at later thresholds; a later loss does not invalidate an earlier break-even. '
            'Use exact arithmetic without intermediate rounding.')


def supplier_question(p):
    lines = [f'Choose the lowest expected-total-cost FEASIBLE plan among the four offered plans for {p["demand"]} units. '
             'Only these plans are allowed. Quantities are booked and paid for in advance; late units have no refund '
             'or salvage value within this decision horizon. A positive order incurs the supplier fee once; zero orders cost zero. '
             'The discounted unit price applies to EVERY ordered unit if its threshold is reached (inclusive). '
             'Supplier capacities are hard booking limits. All amounts are EUR; no other costs or taxes apply.']
    for s in p['suppliers']:
        lines.append(f'{s["name"]}: unit {money(s["unit"])}, discounted unit {money(s["discounted_unit"])} '
                     f'at quantity >= {s["discount_threshold"]}, fee {money(s["fee"])}, capacity {s["capacity"]}.')
    lines.append('The following mutually exclusive JOINT delivery states exhaust all possibilities; do not assume independence. '
                 'Percentages in a state are the fractions of each booked quantity arriving by the common deadline, '
                 'in supplier order S1, S2, S3:')
    for s in p['states']:
        lines.append(f'Probability {s["probability_pct"]}%: on-time percentages {s["on_time_pct"]}.')
    lines.append(f'The probability that at least {p["service_units"]} ORIGINAL units arrive by the deadline must be '
                 f'at least {p["minimum_service_pct"]}% (equality is allowed). '
                 f'Each unit short of the full {p["demand"]} at that deadline incurs an additional emergency cost of '
                 f'EUR {money(p["emergency_unit"])}. Emergency units do NOT count toward the original-delivery service rule. '
                 'Expected total cost = booked purchase cost plus expected emergency cost over the joint states. '
                 'Use exact arithmetic without intermediate rounding.')
    return '\n'.join(lines)


def reference_examples():
    procurement = dict(demand=1000, service_units=1000, minimum_service_pct=90, emergency_unit=2000,
        suppliers=[dict(name='S1', unit=800, discounted_unit=750, discount_threshold=800, fee=20000, capacity=1000),
                   dict(name='S2', unit=700, discounted_unit=680, discount_threshold=800, fee=30000, capacity=1200),
                   dict(name='S3', unit=900, discounted_unit=870, discount_threshold=500, fee=10000, capacity=700)],
        plans=[dict(name='P1', quantities=[1000, 0, 0]), dict(name='P2', quantities=[0, 1000, 0]),
               dict(name='P3', quantities=[500, 0, 500]), dict(name='P4', quantities=[0, 600, 400])],
        states=[dict(probability_pct=70, on_time_pct=[100, 100, 100]),
                dict(probability_pct=20, on_time_pct=[100, 0, 100]),
                dict(probability_pct=10, on_time_pct=[0, 100, 100])])
    production = dict(capacity=160, fixed=60000, price=2000, price_threshold=100, price_cut=100,
                      unit=1200, discount_threshold=110, unit_discount=200, setup_threshold=80,
                      setup=50000, overtime_threshold=120, overtime=200, batch_size=25, batch_fee=3000)
    return procurement, production


def candidate_break_even(rng, level):
    p = dict(capacity=300, fixed=rng.randrange(40000, 180001, 100), price=rng.randrange(1800, 3001, 50),
             unit=rng.randrange(700, 1401, 50), setup_threshold=rng.randrange(40, 91),
             setup=rng.randrange(15000, 70001, 100), price_threshold=rng.randrange(90, 141),
             price_cut=rng.randrange(100, 501, 50) if level == 3 else 0,
             discount_threshold=rng.randrange(90, 151), unit_discount=rng.randrange(100, 401, 50) if level >= 2 else 0,
             overtime_threshold=rng.randrange(100, 161), overtime=rng.randrange(100, 401, 50) if level == 3 else 0,
             batch_size=rng.choice([20, 25, 30, 40]), batch_fee=rng.randrange(1000, 4001, 100) if level == 3 else 0)
    gold = break_even(p)
    active = [p['setup_threshold'] + 1]
    if level >= 2:
        active += [p['discount_threshold']]
    if level == 3:
        active += [p['price_threshold'], p['overtime_threshold'] + 1]
    if gold is None or not max(active) <= gold < p['capacity'] - 2:
        return None
    if gold != break_even_piecewise(p):
        raise AssertionError('Reference disagreement')
    distractors = set()
    for key in ('setup', 'unit_discount', 'price_cut', 'batch_fee', 'overtime'):
        changed = dict(p, **{key: 0})
        other = break_even(changed)
        if other is not None and other != gold:
            distractors.add(other)
    choices = sorted(distractors)
    rng.shuffle(choices)
    for other in (gold - 1, gold + 1, gold + 2):
        if other not in choices:
            choices.append(other)
    return p, gold, [str(x) for x in choices[:3]], break_even_question(p)


def candidate_supplier(rng, level):
    demand = rng.randrange(200, 801, 20)
    suppliers = []
    for i in range(3):
        unit = rng.randrange(500, 1501, 10)
        suppliers.append(dict(name=f'S{i+1}', unit=unit, discounted_unit=unit-rng.randrange(20, 151, 10),
                              discount_threshold=demand*rng.choice([4, 6, 8])//10,
                              fee=rng.randrange(5000, 30001, 100), capacity=demand*rng.choice([8, 10, 12])//10))
    states = [dict(probability_pct=100, on_time_pct=[100, 100, 100])]
    if level == 2:
        a, b = rng.choice([5, 10, 15]), rng.choice([10, 15, 20])
        states = [dict(probability_pct=100-a-b, on_time_pct=[100, 100, 100]),
                  dict(probability_pct=a, on_time_pct=[0, 100, 100]),
                  dict(probability_pct=b, on_time_pct=[100, 0, 100])]
    if level == 3:
        a, b, joint = rng.choice([5, 10, 15]), rng.choice([10, 15, 20]), rng.choice([10, 15, 20])
        states = [dict(probability_pct=100-a-b-joint, on_time_pct=[100, 100, 100]),
                  dict(probability_pct=a, on_time_pct=[0, 100, 100]),
                  dict(probability_pct=b, on_time_pct=[100, 0, 100]),
                  dict(probability_pct=joint, on_time_pct=[0, 0, rng.choice([50, 100])])]
    p = dict(demand=demand, suppliers=suppliers, states=states,
             service_units=demand*rng.choice([5, 7, 9, 10])//10,
             minimum_service_pct=100 if level == 1 else rng.choice([65, 70, 75, 80, 85, 90]),
             emergency_unit=0 if level == 1 else rng.randrange(1000, 4001, 100),
             plans=[dict(name='P1', quantities=[demand, 0, 0]), dict(name='P2', quantities=[0, demand, 0]),
                    dict(name='P3', quantities=[demand//2, 0, demand//2]),
                    dict(name='P4', quantities=[0, demand*6//10, demand*4//10])])
    # Supplier position and plan IDs must not encode a fixed risk role or preferred allocation.
    order = list(range(3))
    rng.shuffle(order)
    p['suppliers'] = [dict(p['suppliers'][old], name=f'S{i+1}') for i, old in enumerate(order)]
    for state in p['states']:
        state['on_time_pct'] = [state['on_time_pct'][old] for old in order]
    rng.shuffle(p['plans'])
    for i, plan in enumerate(p['plans']):
        plan['name'] = f'P{i+1}'
        plan['quantities'] = [plan['quantities'][old] for old in order]
    try:
        gold = supplier_gold(p)
    except ValueError:
        return None
    return p, gold, [x['name'] for x in p['plans'] if x['name'] != gold], supplier_question(p)


def generate(seed=DEFAULT_SEED, per_cell=4):
    if per_cell <= 0 or per_cell % 4:
        raise ValueError('per_cell must be a positive multiple of four for label balance')
    rng = random.Random(seed)
    tasks, seen = [], set()
    for family in ('break_even', 'supplier'):
        for level in (1, 2, 3):
            labels = list('ABCD') * (per_cell // 4)
            rng.shuffle(labels)
            for index, label in enumerate(labels):
                candidate, rejected = None, 0
                while candidate is None:
                    if rejected >= 10000:
                        raise RuntimeError('Generator rejection limit reached')
                    candidate = (candidate_break_even if family == 'break_even' else candidate_supplier)(rng, level)
                    rejected += 1
                    if candidate is not None and candidate[3] in seen:
                        candidate = None
                p, gold, distractors, question = candidate
                seen.add(question)
                rng.shuffle(distractors)
                values = {k: str(gold) if k == label else str(distractors.pop()) for k in 'ABCD'}
                options = dict(values)
                if family == 'supplier':
                    plans = {x['name']: x['quantities'] for x in p['plans']}
                    options = {k: f'{v}: quantities S1/S2/S3 = {plans[v]}' for k, v in values.items()}
                tasks.append(dict(id=f'v2_{seed}_{family}_L{level}_{index+1:02}', family=family,
                                  template=f'{family}_L{level}', complexity=level, split='development',
                                  generator_version=VERSION, question=question, options=options, answer=label,
                                  reference=dict(parameters=p, value=gold, option_values=values,
                                                 rejected_before_acceptance=rejected-1)))
    validate(tasks)
    return tasks


def validate(tasks):
    if not tasks or len({t['id'] for t in tasks}) != len(tasks) or len({t['question'] for t in tasks}) != len(tasks):
        raise ValueError('Empty or duplicate dataset')
    cells = {}
    for t in tasks:
        if len(set(t['options'].values())) != 4 or set(t['options']) != set('ABCD'):
            raise ValueError('Four distinct options required')
        p = t['reference']['parameters']
        if t['family'] == 'break_even':
            gold = break_even(p)
            if gold != break_even_piecewise(p):
                raise ValueError('Independent reference disagreement')
            if t['options'] != t['reference']['option_values']:
                raise ValueError('Option rendering disagreement')
        elif t['family'] == 'supplier':
            first = [(s['name'], s['feasible'], s['expected_cost_cents'], s['service_probability']) for s in supplier_scores(p)]
            if first != supplier_scores_enumerated(p):
                raise ValueError('Independent reference disagreement')
            gold = supplier_gold(p)
            plans = {x['name']: x['quantities'] for x in p['plans']}
            expected_options = {k: f'{v}: quantities S1/S2/S3 = {plans[v]}' for k, v in t['reference']['option_values'].items()}
            if t['options'] != expected_options:
                raise ValueError('Option rendering disagreement')
        else:
            raise ValueError('Unknown family')
        expected_question = (break_even_question if t['family'] == 'break_even' else supplier_question)(p)
        if t['question'] != expected_question:
            raise ValueError('Prompt and reference parameter disagreement')
        if str(gold) != t['reference']['option_values'][t['answer']] or gold != t['reference']['value']:
            raise ValueError('Wrong gold label')
        cells.setdefault((t['family'], t['complexity']), Counter())[t['answer']] += 1
    if len(cells) != 6 or any(set(c) != set('ABCD') or len(set(c.values())) != 1 for c in cells.values()):
        raise ValueError('All six cells must have balanced labels')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--per-cell', type=int, default=4)
    args = parser.parse_args()
    tasks = generate(args.seed, args.per_cell)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)
    print(f'{len(tasks)} development tasks; both reference implementations agree.')


if __name__ == '__main__':
    main()
