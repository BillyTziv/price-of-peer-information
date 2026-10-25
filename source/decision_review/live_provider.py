"""Pinned text-only OpenAI adapter with conservative pre-dispatch reservation."""
import json
import math
import os

MODELS = {'small': 'gpt-4.1-mini-2025-04-14', 'strong': 'gpt-4.1-2025-04-14'}
RATES = {'small': (0.40, 0.10, 1.60), 'strong': (2.00, 0.50, 8.00)}
CONTEXT = 1047576
SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'decision', 'strict': True,
          'schema': {'type': 'object', 'properties': {'rationale': {'type': 'string'},
          'answer': {'type': 'string', 'enum': ['A', 'B', 'C', 'D', 'ABSTAIN']}},
          'required': ['rationale', 'answer'], 'additionalProperties': False}}}


class LiveError(RuntimeError):
    def __init__(self, message, record):
        super().__init__(message)
        self.record = record


class OpenAIProvider:
    def __init__(self, budget_eur, eur_per_usd, fx_date, max_output=1024, temperature=1.0,
                 transport=None, timeout_seconds=60):
        if any(not math.isfinite(x) or x <= 0 for x in (budget_eur, eur_per_usd)):
            raise ValueError('Budget and EUR per USD must be finite and positive')
        if not 1 <= max_output <= 32768 or not 0 <= temperature <= 2:
            raise ValueError('Invalid generation settings')
        if not math.isfinite(timeout_seconds) or not 1 <= timeout_seconds <= 600:
            raise ValueError('Invalid API timeout')
        from datetime import date
        date.fromisoformat(fx_date)
        self.budget, self.fx, self.fx_date = budget_eur, eur_per_usd, fx_date
        self.max_output, self.temperature, self.timeout_seconds = max_output, temperature, timeout_seconds
        self.spent, self.unknown_reserved, self.attempts = 0.0, 0.0, 0
        self.halted = False
        if transport is None:
            if not os.environ.get('OPENAI_API_KEY'):
                raise ValueError('Set OPENAI_API_KEY locally; never paste it into chat')
            from openai import OpenAI
            client = OpenAI(max_retries=0, timeout=float(timeout_seconds), base_url='https://api.openai.com/v1')
            transport = lambda body: client.chat.completions.create(**body).model_dump()
        self.transport = transport

    def metadata(self):
        return dict(provider='openai', models=MODELS, rates_usd_per_million=RATES,
                    rates_checked='2026-10-10', eur_per_usd=self.fx, fx_date=self.fx_date,
                    budget_eur=self.budget, max_output_tokens=self.max_output,
                    temperature=self.temperature, top_p=1.0, retries=0,
                    timeout_seconds=self.timeout_seconds,
                    reservation='full model context at uncached input rate plus maximum output',
                    known_physical_cost_eur=self.spent, unknown_reserved_eur=self.unknown_reserved,
                    dispatched_attempts=self.attempts, halted=self.halted)

    def __call__(self, model, request, call_id):
        if self.halted:
            raise LiveError('Provider halted', {'dispatched': False, 'cost_unknown': False})
        inp, cached_rate, output = RATES[model]
        # Bound by model capacity, not a guessed tokenizer estimate. Deliberately conservative.
        reserve = (CONTEXT * inp + self.max_output * output) / 1e6 * self.fx
        if self.spent + self.unknown_reserved + reserve > self.budget:
            raise LiveError('Insufficient budget for next worst-case call',
                            {'dispatched': False, 'cost_unknown': False, 'reserved_eur': reserve})
        body = dict(model=MODELS[model], messages=[{'role': 'system', 'content': request['system']},
                    {'role': 'user', 'content': json.dumps({k:v for k,v in request.items() if k != 'system'}, ensure_ascii=False)}],
                    response_format=SCHEMA, max_completion_tokens=self.max_output,
                    temperature=self.temperature, top_p=1.0, store=False, service_tier='default')
        rec = dict(dispatched=True, api_request=body, reserved_eur=reserve, cost_unknown=True)
        self.attempts += 1
        self.unknown_reserved += reserve
        try:
            raw = self.transport(body)
            rec['raw_response'] = raw
            u = raw['usage']
            p, c = u['prompt_tokens'], u['completion_tokens']
            cached = (u.get('prompt_tokens_details') or {}).get('cached_tokens', 0)
            if any(type(v) is not int or v < 0 for v in (p, c, cached)) or cached > p:
                raise ValueError('Invalid usage')
            cost = ((p-cached)*inp + cached*cached_rate + c*output) / 1e6 * self.fx
            self.spent += cost
            self.unknown_reserved -= reserve
            rec.update(cost_eur=cost, cost_unknown=False, usage=u, resolved_model=raw['model'])
            if raw['model'] != MODELS[model]:
                raise ValueError('Resolved model differs from pinned snapshot')
            if p > CONTEXT or c > self.max_output or cost > reserve:
                raise ValueError('Usage exceeds reserved bound')
            choice = raw['choices'][0]
            message = choice['message']
            rec.update(finish_reason=choice['finish_reason'], refusal=message.get('refusal'))
            payload = {}
            if choice['finish_reason'] == 'stop' and not message.get('refusal'):
                try:
                    payload = json.loads(message.get('content') or '')
                except (ValueError, TypeError):
                    pass
            if not isinstance(payload, dict):
                payload = {}
            valid = (isinstance(payload.get('rationale'), str) and
                     payload.get('answer') in ('A','B','C','D','ABSTAIN'))
            rec.update(rationale=payload['rationale'] if valid else '',
                       answer=payload['answer'] if valid else 'INVALID')
            return rec
        except Exception as exc:
            self.halted = True
            rec['error_type'] = type(exc).__name__
            raise LiveError('API call failed; inspect retained logs before restarting', rec) from exc
