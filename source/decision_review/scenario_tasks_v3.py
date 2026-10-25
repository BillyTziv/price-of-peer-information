"""One prespecified, generator-wide revision. DEVELOPMENT ONLY; no model calls.

L1 retains the v2 control mechanism with fresh parameters. L2/L3 introduce
pack-based production costs and state-dependent procurement recourse.
Amounts are integer cents; probability-weighted amounts use exact Fraction.
"""
import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random

from . import scenario_tasks as v2

VERSION = 'economic-scenarios-v3.0'
SCREEN_SEED = 2026101201
CONFIRMATION_SEED = 2026101202


def ceiling(n, d):
    return (n + d - 1) // d


def production_profit(p, q):
    """Both components are purchased in whole packs; unused inputs have no value."""
    revenue = q * (p['price'] - (p['price_cut'] if q >= p['price_threshold'] else 0))
    materials = 0
    for component in p['components']:
        packs = ceiling(q, component['pack_size'])
        unit = (component['discounted_unit'] if packs >= component['discount_packs']
                else component['unit'])
        materials += packs * component['pack_size'] * unit
    return (revenue - materials - p['fixed']
            - (p['setup'] if q > p['setup_threshold'] else 0)
            - p['overtime'] * max(q - p['overtime_threshold'], 0)
            - p['batch_fee'] * ceiling(q, p['batch_size']))


def production_brute(p):
    return next((q for q in range(p['capacity'] + 1) if production_profit(p, q) >= 0), None)


def production_intervals(p):
    """Independent affine-interval solver; never calls production_profit.

    Enumerates every material/shipping discontinuity and every rule threshold.
    Inside an interval, pack costs are constant and revenue/overtime are linear.
    """
    points = {0, p['capacity'] + 1, p['price_threshold'], p['setup_threshold'] + 1,
              p['overtime_threshold'] + 1}
    for size in [p['batch_size']] + [c['pack_size'] for c in p['components']]:
        points.update(range(1, p['capacity'] + 1, size))
    points = sorted(x for x in points if 0 <= x <= p['capacity'] + 1)
    for lo, end in zip(points, points[1:]):
        price = p['price'] - (p['price_cut'] if lo >= p['price_threshold'] else 0)
        overtime = p['overtime'] if lo > p['overtime_threshold'] else 0
        costs = p['fixed'] + (p['setup'] if lo > p['setup_threshold'] else 0)
        for c in p['components']:
            packs = lo // c['pack_size'] + int(lo % c['pack_size'] != 0)
            rate = c['unit'] if packs < c['discount_packs'] else c['discounted_unit']
            costs += packs * c['pack_size'] * rate
        costs += (lo // p['batch_size'] + int(lo % p['batch_size'] != 0)) * p['batch_fee']
        intercept = -costs + overtime * p['overtime_threshold']
        slope = price - overtime
        if slope * lo + intercept >= 0:
            return lo
        if slope > 0:
            q = max(lo, -(intercept // slope))
            if q < end:
                return q
    return None


def profit_islands(p):
    starts, previous = [], False
    for q in range(p['capacity'] + 1):
        positive = production_profit(p, q) >= 0
        if positive and not previous:
            starts.append(q)
        previous = positive
    return starts


def procurement_scores(p):
    """Two-period stock flow with one emergency pool shared across both periods."""
    output = []
    for plan in p['plans']:
        qs = plan['quantities']
        base = sum(q * (s['discounted_unit'] if q >= s['discount_threshold'] else s['unit'])
                   + (s['fee'] if q else 0) for q, s in zip(qs, p['suppliers']))
        booking_ok = (sum(qs) == sum(p['demands'])
                      and all(0 <= q <= s['capacity'] for q, s in zip(qs, p['suppliers'])))
        expected, service, ledger = Fraction(0), Fraction(0), []
        for state in p['states']:
            receipts = [sum(Fraction(q * pc, 100) for q, pc in zip(qs, percentages))
                        for percentages in state['cumulative_pct']]
            available1 = receipts[0]
            e1 = min(max(p['demands'][0] - available1, 0), p['reserve'])
            lost1 = max(p['demands'][0] - available1 - e1, 0)
            carry = max(available1 - p['demands'][0], 0)
            available2 = carry + receipts[1] - receipts[0]
            e2 = min(max(p['demands'][1] - available2, 0), p['reserve'] - e1)
            lost2 = max(p['demands'][1] - available2 - e2, 0)
            leftover = max(available2 - p['demands'][1], 0)
            costs = (base + p['emergency_unit'] * (e1 + e2)
                     + p['activation_fee'] * (int(e1 > 0) + int(e2 > 0))
                     + p['lost_unit'] * (lost1 + lost2) + p['holding_unit'] * carry
                     - p['salvage_unit'] * leftover)
            success = lost1 == lost2 == 0
            probability = Fraction(state['probability_pct'], 100)
            expected += probability * costs
            service += probability * success
            ledger.append(dict(receipts=receipts, emergency=[e1, e2], lost=[lost1, lost2],
                               carry=carry, leftover=leftover, cost_cents=costs, success=success))
        output.append(dict(name=plan['name'], base_cents=base, expected_cost_cents=expected,
                           service_probability=service,
                           feasible=booking_ok and service >= Fraction(p['minimum_service_pct'], 100),
                           states=ledger))
    return output


def procurement_simulated(p):
    """Independent unit-ledger simulation on 100 equally likely tickets.

    Unit arrivals, customers, emergency availability and remaining inventory
    are processed in time order; no aggregate flow formulas are reused.
    """
    scores = []
    for plan in p['plans']:
        qs = plan['quantities']
        booked = 0
        for q, supplier in zip(qs, p['suppliers']):
            rate = supplier['unit']
            if q >= supplier['discount_threshold']:
                rate = supplier['discounted_unit']
            booked += sum(rate for _ in range(q))
            if q > 0:
                booked += supplier['fee']
        costs, successes = [], 0
        for state in p['states']:
            arrivals = [[], []]
            for supplier_index, quantity in enumerate(qs):
                cumulative = [state['cumulative_pct'][k][supplier_index] for k in (0, 1)]
                if any(quantity * pct % 100 for pct in cumulative):
                    raise ValueError('Integer delivered units required')
                for unit in range(1, quantity + 1):
                    if unit <= quantity * cumulative[0] // 100:
                        arrivals[0].append(1)
                    elif unit <= quantity * cumulative[1] // 100:
                        arrivals[1].append(1)
            cash, inventory, spare, complete = booked, [], [1] * p['reserve'], True
            for period in (0, 1):
                inventory.extend(arrivals[period])
                activated = False
                for _ in range(p['demands'][period]):
                    if inventory:
                        inventory.pop()
                    elif spare:
                        spare.pop()
                        cash += p['emergency_unit']
                        if not activated:
                            cash += p['activation_fee']
                            activated = True
                    else:
                        cash += p['lost_unit']
                        complete = False
                if period == 0:
                    cash += sum(p['holding_unit'] for _ in inventory)
                else:
                    cash -= sum(p['salvage_unit'] for _ in inventory)
            costs.extend([cash] * state['probability_pct'])
            successes += int(complete) * state['probability_pct']
        if len(costs) != 100:
            raise ValueError('Probabilities must sum to 100')
        booking_ok = (sum(qs) == sum(p['demands'])
                      and all(0 <= q <= s['capacity'] for q, s in zip(qs, p['suppliers'])))
        scores.append((plan['name'], booking_ok and successes >= p['minimum_service_pct'],
                       Fraction(sum(costs), 100), Fraction(successes, 100)))
    return scores


def procurement_gold(p):
    ranked = sorted((s['expected_cost_cents'], s['name']) for s in procurement_scores(p) if s['feasible'])
    if len(ranked) < 2 or ranked[0][0] == ranked[1][0]:
        raise ValueError('At least two feasible plans and a unique best required')
    return ranked[0][1]


def production_question(p):
    lines = [f'A producer makes and sells exactly q finished units, integer 0 <= q <= {p["capacity"]}. '
             'Each finished unit consumes one unit of EACH of two different components. '
             'Components are purchased in whole packs separately; unused components have zero recovery value. '
             f'Revenue is EUR {v2.money(p["price"])} per finished unit, reduced by EUR {v2.money(p["price_cut"])} '
             f'on EVERY unit when q >= {p["price_threshold"]}.']
    for i, c in enumerate(p['components'], 1):
        lines.append(f'Component {i}: pack size {c["pack_size"]}, price EUR {v2.money(c["unit"])} per COMPONENT UNIT '
                     f'(not per pack); if at least {c["discount_packs"]} packs are bought, the price becomes '
                     f'EUR {v2.money(c["discounted_unit"])} for EVERY purchased unit of that component.')
    lines.append(f'Fixed cost is EUR {v2.money(p["fixed"])} even at q=0. Setup adds EUR {v2.money(p["setup"])} '
                 f'once when q > {p["setup_threshold"]}. Overtime adds EUR {v2.money(p["overtime"])} '
                 f'for each finished unit ABOVE {p["overtime_threshold"]}. Shipping costs EUR {v2.money(p["batch_fee"])} '
                 f'per started batch of {p["batch_size"]} finished units. At q=0 buy no packs and ship no batches. '
                 'No other costs, revenues or taxes. Find the SMALLEST q with revenue at least total cost. '
                 'Profit need not be monotone; an earlier profitable quantity remains a valid break-even even if a later '
                 'quantity loses money. Use exact arithmetic. Keep the rationale concise, stating key boundary calculations.')
    return '\n'.join(lines)


def procurement_question(p):
    lines = [f'Choose the lowest expected-total-cost FEASIBLE one of the four offered procurement plans. '
             f'Demand at deadline 1 is {p["demands"][0]} units; at deadline 2 it is {p["demands"][1]} units. '
             'Book and pay all supplier orders before deadline 1. The same interchangeable product is supplied. '
             'A positive order incurs its supplier fee once; a zero order has no fee. '
             'Discounted prices apply to EVERY booked unit when the inclusive threshold is met. '
             'All amounts are EUR; no other cash flows or taxes apply.']
    for s in p['suppliers']:
        lines.append(f'{s["name"]}: unit {v2.money(s["unit"])}, discounted unit {v2.money(s["discounted_unit"])} '
                     f'at q >= {s["discount_threshold"]}, fee {v2.money(s["fee"])}, capacity {s["capacity"]}.')
    lines.append('The following mutually exclusive JOINT states exhaust all possibilities. Percentages are CUMULATIVE '
                 'fractions of each booked supplier quantity received by deadlines 1 and 2, in order S1/S2/S3. '
                 'Do not add cumulative receipts together or assume independent suppliers.')
    for s in p['states']:
        lines.append(f'{s["probability_pct"]}%: deadline 1 {s["cumulative_pct"][0]}; deadline 2 {s["cumulative_pct"][1]}.')
    lines.append(f'At each deadline use available regular stock first. Immediately cover any remaining demand with as '
                 f'many emergency units as possible from one pool of {p["reserve"]} units shared across BOTH deadlines. '
                 f'Only used emergency units are paid, at EUR {v2.money(p["emergency_unit"])} each; each deadline using '
                 f'any emergency supply also incurs a single activation fee EUR {v2.money(p["activation_fee"])}. '
                 f'After this mandatory policy, each unmet unit is a lost sale costing EUR {v2.money(p["lost_unit"])} '
                 '(the entire lost-sale consequence; add no other lost revenue). Lost demand is not backlogged. '
                 f'Regular stock remaining after deadline 1 is carried to deadline 2 for EUR {v2.money(p["holding_unit"])} '
                 f'per unit. Regular stock left after deadline 2 earns salvage EUR {v2.money(p["salvage_unit"])} per unit. '
                 'Emergency supply is bought only for immediate shortage and is never carried. Booked units that have '
                 'not arrived by deadline 2 have no refund or salvage. '
                 f'A plan is feasible only if capacities are respected and the probability of ZERO lost sales at BOTH '
                 f'deadlines is at least {p["minimum_service_pct"]}% (inclusive). Emergency supply DOES count toward '
                 'this service test. Expected total cost includes booked cost, emergency purchases and activation, '
                 'lost-sale penalties and holding, minus salvage. Use exact arithmetic. '
                 'Keep the rationale concise: state each plan\'s service probability and expected cost.')
    return '\n'.join(lines)


def production_candidate(rng, level):
    components = []
    for sizes in ([7, 9, 11, 13], [8, 12, 16, 18]):
        unit = rng.randrange(200, 701, 25)
        components.append(dict(pack_size=rng.choice(sizes), unit=unit,
                               discounted_unit=unit-(rng.choice([25, 50, 75]) if level == 3 else 0),
                               discount_packs=rng.randint(6, 14)))
    p = dict(capacity=240, components=components, fixed=rng.randrange(25000, 110001, 100),
             price=sum(c['unit'] for c in components)+rng.randrange(500, 1301, 50),
             price_cut=rng.choice([100, 150, 200, 250]) if level == 3 else 0,
             price_threshold=rng.randint(60, 150), setup=rng.randrange(10000, 40001, 100),
             setup_threshold=rng.randint(35, 100), overtime=rng.choice([100, 150, 200]) if level == 3 else 0,
             overtime_threshold=rng.randint(100, 170), batch_size=rng.choice([17, 23, 29]),
             batch_fee=rng.randrange(1500, 4501, 100))
    starts = profit_islands(p)
    if len(starts) < (2 if level == 2 else 3):
        return None
    gold = starts[0]
    if not 25 <= gold <= 180 or production_profit(p, p['capacity']) < 0:
        return None
    # Pack purchasing must change the decision; continuous material costing is a distractor.
    continuous = deepcopy(p)
    for c in continuous['components']:
        c['discount_packs'] = (c['discount_packs']-1)*c['pack_size']+1
        c['pack_size'] = 1
    relaxed = production_brute(continuous)
    if relaxed is None or relaxed >= gold:
        return None
    # L3 must actually expose an all-unit price/discount threshold before the first break-even.
    if level == 3 and not (p['price_threshold'] <= gold or any(
            ceiling(gold, c['pack_size']) >= c['discount_packs'] for c in p['components'])):
        return None
    distractors = [relaxed, starts[1]]
    third = starts[2] if len(starts) >= 3 else gold-1
    if third in distractors or third == gold:
        third = gold+1
    return p, gold, list(map(str, distractors+[third])), production_question(p), dict(
        profit_island_starts=starts, continuous_pack_relaxation=relaxed,
        nonmonotone=True)


def procurement_candidate(rng, level):
    total = rng.choice([120, 160, 200, 240])
    d1 = total if level == 2 else total*rng.choice([4, 5, 6])//10
    suppliers = []
    for i in range(3):
        unit = rng.randrange(500, 1301, 25)
        suppliers.append(dict(name=f'S{i+1}', unit=unit, discounted_unit=unit-rng.choice([25, 50, 75, 100]),
                              discount_threshold=total*rng.choice([3, 4, 5])//10,
                              fee=rng.randrange(1000, 12001, 100), capacity=total))
    # Mixed allocations make all four options comply with booking capacity.
    grid = [[a, b, 10-a-b] for a in range(1, 9) for b in range(1, 10-a)]
    allocations = rng.sample(grid, 4)
    plans = [dict(name=f'P{i+1}', quantities=[total*x//10 for x in shares]) for i, shares in enumerate(allocations)]
    if level == 2:
        delivery = [[[100,100,100],[100,100,100]], [[0,100,100],[0,100,100]],
                    [[100,0,100],[100,0,100]], [[50,50,100],[50,50,100]]]
    else:
        delivery = [[[100,50,0],[100,100,100]], [[0,50,50],[50,100,100]],
                    [[50,0,50],[100,50,100]], [[50,50,0],[50,100,50]]]
    weights = rng.choice([[55,15,20,10],[60,20,10,10],[50,20,15,15],[65,10,15,10]])
    # Randomize supplier risk roles, not just their display labels.
    permutation = list(range(3)); rng.shuffle(permutation)
    states = [dict(probability_pct=w,cumulative_pct=[[v[i] for i in permutation] for v in matrix])
              for w, matrix in zip(weights, delivery)]
    rng.shuffle(states)
    p = dict(demands=[d1,total-d1], suppliers=suppliers, plans=plans, states=states,
             reserve=total*rng.choice([2,3,4,5])//10, emergency_unit=rng.randrange(1800,3501,100),
             activation_fee=rng.randrange(2000,10001,100), lost_unit=rng.randrange(4000,8001,100),
             holding_unit=rng.randrange(25,201,25) if level == 3 else 0,
             salvage_unit=rng.randrange(100,401,25) if level == 3 else 0,
             minimum_service_pct=rng.choice([65,70,75,80,85]))
    scores = procurement_scores(p)
    ranked = sorted((s['expected_cost_cents'], s['name']) for s in scores if s['feasible'])
    if len(ranked) < 2 or ranked[0][0] == ranked[1][0]:
        return None
    cheapest = min(scores, key=lambda s: (s['base_cents'],s['name']))
    if (sum(s['base_cents']==cheapest['base_cents'] for s in scores)!=1
            or not cheapest['feasible'] or cheapest['name'] == ranked[0][1]):
        return None
    gap = (ranked[1][0]-ranked[0][0])/ranked[0][0]
    if not Fraction(1,200) <= gap <= Fraction(3,20):
        return None
    if level == 3:
        # Inventory carry and exhaustion of the common emergency pool must occur.
        ledgers = [s for plan in scores for s in plan['states']]
        if not any(s['carry'] > 0 for s in ledgers) or not any(
                s['emergency'][0] > 0 and s['lost'][1] > 0 for s in ledgers):
            return None
    gold = ranked[0][1]
    return p, gold, [x['name'] for x in plans if x['name']!=gold], procurement_question(p), dict(
        cheapest_booked_plan=cheapest['name'], feasible_plans=len(ranked),
        relative_gap_to_second=float(gap), all_plans_within_booking_capacity=True)


def render_options(kind, p, values):
    if kind in ('procurement_recourse','supplier_control'):
        quantities = {x['name']:x['quantities'] for x in p['plans']}
        return {k:f'{name}: quantities S1/S2/S3 = {quantities[name]}' for k,name in values.items()}
    return dict(values)


def generate(seed=SCREEN_SEED, per_cell=4):
    if per_cell <= 0 or per_cell % 4:
        raise ValueError('per_cell must be a positive multiple of four')
    rng = random.Random(seed)
    tasks, seen = [], set()
    for family in ('break_even','supplier'):
        for level in (1,2,3):
            labels = list('ABCD')*(per_cell//4); rng.shuffle(labels)
            numeric_ranks = list(range(4))*(per_cell//4); rng.shuffle(numeric_ranks)
            for index,label in enumerate(labels,1):
                for rejected in range(20000):
                    if level == 1:
                        old = (v2.candidate_break_even if family=='break_even' else v2.candidate_supplier)(rng,1)
                        candidate = (*old, {'fresh_v2_control':True}) if old else None
                        kind = 'production_control' if family=='break_even' else 'supplier_control'
                    else:
                        candidate = (production_candidate if family=='break_even' else procurement_candidate)(rng,level)
                        kind = 'pack_production' if family=='break_even' else 'procurement_recourse'
                    if candidate is not None and candidate[3] not in seen:
                        break
                else:
                    raise RuntimeError('Generator rejection limit reached')
                p,gold,distractors,question,features = candidate
                if family=='break_even':
                    # Balance the gold's numeric rank separately from its A/B/C/D position.
                    # Otherwise "choose the second-smallest number" could leak the answer.
                    below = [int(x) for x in distractors if int(x)<gold]
                    above = [int(x) for x in distractors if int(x)>gold]
                    below = list(dict.fromkeys(below+[gold-i for i in (1,2,3)]))
                    above = list(dict.fromkeys(above+[gold+i for i in (1,2,3)]))
                    n_below = numeric_ranks[index-1]
                    distractors = list(map(str,below[:n_below]+above[:3-n_below]))
                    features = dict(features,gold_numeric_rank=n_below+1)
                else:
                    # Balance semantic plan IDs independently of answer-label positions.
                    desired = f'P{numeric_ranks[index-1]+1}'
                    old_others = [x['name'] for x in p['plans'] if x['name']!=gold]
                    new_others = [f'P{i}' for i in range(1,5) if f'P{i}'!=desired]
                    rng.shuffle(new_others)
                    names = dict(zip(old_others,new_others)); names[gold]=desired
                    for plan in p['plans']:
                        plan['name']=names[plan['name']]
                    if 'cheapest_booked_plan' in features:
                        features['cheapest_booked_plan']=names[features['cheapest_booked_plan']]
                    gold = desired
                    distractors = new_others
                seen.add(question); rng.shuffle(distractors)
                values = {k:str(gold) if k==label else str(distractors.pop()) for k in 'ABCD'}
                tasks.append(dict(id=f'v3_{seed}_{family}_L{level}_{index:02}', family=family,
                    template=f'{family}_v3_L{level}', complexity=level, split='development',
                    generator_version=VERSION, question=question, options=render_options(kind,p,values), answer=label,
                    reference=dict(kind=kind,parameters=p,value=gold,option_values=values,
                                   rejected_before_acceptance=rejected,features=features)))
    validate(tasks)
    return tasks


def validate(tasks):
    if not tasks or len({t['id'] for t in tasks}) != len(tasks) or len({t['question'] for t in tasks}) != len(tasks):
        raise ValueError('Empty or duplicate dataset')
    cells, ranks, plan_ids = {}, {}, {}
    for t in tasks:
        if t['generator_version']!=VERSION or t['split']!='development' or t['complexity'] not in (1,2,3):
            raise ValueError('Unsupported development metadata')
        p,kind = t['reference']['parameters'],t['reference']['kind']
        if kind=='production_control':
            gold,other,question = v2.break_even(p),v2.break_even_piecewise(p),v2.break_even_question(p)
        elif kind=='supplier_control':
            scores = v2.supplier_scores(p)
            if [(s['name'],s['feasible'],s['expected_cost_cents'],s['service_probability']) for s in scores] != v2.supplier_scores_enumerated(p):
                raise ValueError('Control reference disagreement')
            gold = other = v2.supplier_gold(p); question = v2.supplier_question(p)
        elif kind=='pack_production':
            gold,other,question = production_brute(p),production_intervals(p),production_question(p)
        elif kind=='procurement_recourse':
            if any(len(s['cumulative_pct'])!=2 or not all(0<=a<=b<=100 for a,b in zip(*s['cumulative_pct'])) for s in p['states']):
                raise ValueError('Invalid cumulative receipts')
            scores = procurement_scores(p)
            if [(s['name'],s['feasible'],s['expected_cost_cents'],s['service_probability']) for s in scores] != procurement_simulated(p):
                raise ValueError('Procurement reference disagreement')
            gold = other = procurement_gold(p); question = procurement_question(p)
        else:
            raise ValueError('Unknown reference kind')
        expected_kind = ('production_control' if t['complexity']==1 else 'pack_production') if t['family']=='break_even' else ('supplier_control' if t['complexity']==1 else 'procurement_recourse')
        if t['family'] not in ('break_even','supplier') or kind!=expected_kind:
            raise ValueError('Family/level and reference kind disagree')
        values = t['reference']['option_values']
        if (gold is None or gold!=other or str(gold)!=str(t['reference']['value']) or
                values.get(t['answer'])!=str(gold) or t['question']!=question or
                set(values)!=set('ABCD') or len(set(values.values()))!=4 or
                t['options']!=render_options(kind,p,values)):
            raise ValueError('Reference, question or options mismatch')
        cells.setdefault((t['family'],t['complexity']),Counter())[t['answer']]+=1
        if t['family']=='break_even':
            rank = sorted(map(int,values.values())).index(int(gold))+1
            ranks.setdefault(t['complexity'],Counter())[rank]+=1
        else:
            plan_ids.setdefault(t['complexity'],Counter())[gold]+=1
    if len(cells)!=6 or any(set(c)!=set('ABCD') or len(set(c.values()))!=1 for c in cells.values()):
        raise ValueError('All six cells must have balanced labels')
    if any(set(c)!={1,2,3,4} or len(set(c.values()))!=1 for c in ranks.values()):
        raise ValueError('Production gold numeric ranks must also be balanced')
    if any(set(c)!={'P1','P2','P3','P4'} or len(set(c.values()))!=1 for c in plan_ids.values()):
        raise ValueError('Procurement gold plan IDs must also be balanced')


def reference_examples():
    production = dict(capacity=30,price=1200,price_cut=100,price_threshold=12,
        components=[dict(pack_size=4,unit=200,discounted_unit=200,discount_packs=99),
                    dict(pack_size=3,unit=100,discounted_unit=100,discount_packs=99)],
        fixed=3200,setup=6000,setup_threshold=8,overtime=100,overtime_threshold=20,
        batch_size=5,batch_fee=300)
    procurement = dict(demands=[40,60],reserve=20,emergency_unit=1200,activation_fee=1000,
        lost_unit=4000,holding_unit=100,salvage_unit=200,minimum_service_pct=75,
        suppliers=[dict(name=f'S{i+1}',unit=rate,discounted_unit=rate,discount_threshold=100,fee=0,capacity=100)
                   for i,rate in enumerate([500,700,900])],
        plans=[dict(name=f'P{i+1}',quantities=q) for i,q in enumerate([[60,20,20],[20,60,20],[20,20,60],[40,40,20]])],
        states=[dict(probability_pct=50,cumulative_pct=[[100,50,0],[100,100,100]]),
                dict(probability_pct=25,cumulative_pct=[[0,50,50],[50,100,100]]),
                dict(probability_pct=25,cumulative_pct=[[0,0,50],[50,50,100]])])
    return production,procurement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=SCREEN_SEED)
    parser.add_argument('--per-cell',type=int,default=4)
    a = parser.parse_args()
    tasks = generate(a.seed,a.per_cell)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf-8') as f:
        json.dump(tasks,f,ensure_ascii=False,indent=2)
    print(f'{len(tasks)} audited development tasks: {a.out}')


if __name__=='__main__':
    main()
