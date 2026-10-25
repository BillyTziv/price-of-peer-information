# The Price of Peer Information

Research materials for **The Price of Peer Information: Accuracy, Cost, and Unanimous Errors in LLM Agent Teams**.

Three agents first answer independently, then revise once. In **IR3**, each agent reviews its own response. In **D3**, each agent also sees the other two agents' initial answers and reasoning. Both approaches share the initial responses and use the same voting rule.

## Explore the questions

Download and open [`questions/QUESTIONS.html`](questions/QUESTIONS.html) in a browser to search all **192 exact questions** by family and complexity. Each question has four options and a collapsed reference answer. The catalogue works offline.

[`evidence/main_dataset.json`](evidence/main_dataset.json) is the complete machine-readable dataset: question text, options, answer key, reference calculation parameters and generator metadata. Reference answers and generator metadata were not provided to the models. There are two task families (break-even and supplier selection), three complexity levels, and 32 tasks per family-level combination. Each task was evaluated three times, giving 576 paired evaluations.

## Verify the main results

With Python 3.10 or newer, run from this directory:

```bash
python verify_results.py
```

This uses only the Python standard library. **No API key, internet connection or paid model calls are required.**

The script regenerates and validates the dataset, checks final voting from retained labels, recomputes the primary task-cluster bootstrap interval, and checks the mean exploratory SC6 differences. It prints a JSON report and fails if a check does not match the retained results.

| Outcome | IR3 | D3 |
| --- | ---: | ---: |
| Correct decisions | 401 / 576 | 431 / 576 |
| Accuracy | 69.62% | 74.83% |
| Wrong decisions | 122 | 138 |
| No answer | 53 | 7 |
| Unanimously wrong teams | 42 | 99 |

Primary accuracy difference: **+5.21 percentage points**, with a task-cluster bootstrap 95% interval of **2.95–7.64**. Attributed cost is 24.1% higher for D3; this is a retained cost summary, not a token-log audit performed by the verification script. Greater agreement alone does not establish greater reliability.

## Files

| Path | Contents |
| --- | --- |
| `questions/QUESTIONS.html` | Searchable exact question catalogue |
| `evidence/main_dataset.json` | All questions, options and reference answers |
| `evidence/paired_transitions.csv` | 576 paired final-outcome records and final agent labels; repetitions numbered 0–2 |
| `evidence/reproduced-main-analysis.json` | Retained main analysis, including the six original methods and cost summaries |
| `evidence/task_level_comparison.csv` | Task-level exploratory SC6 comparisons |
| `evidence/six_sample_comparison.json` | Retained SC6 summary and exploratory intervals |
| `source/decision_review/` | Main experiment, generator, reference calculations, prompts, voting and provider implementation |
| `source/requirements-live.txt` | Dependencies for optional live execution; unnecessary for offline checks |
| `verify_results.py` | Offline verification entry point |

## Scope and provenance

This is a focused source export of the study. Unrelated pilot entry points and project-navigation comments were removed; the main-study implementation, generators, reference calculations and retained data are unchanged. The shared aggregation helpers and paired runner used by the study are preserved. `verify_results.py` is a later verification wrapper, not part of the original frozen analysis.

The canonical dataset SHA-256 is `f9138a155c8f9ee173f163c69c4ee933eb1b88bbd50adb9309a9b101147963e3` (sorted-key JSON with Unicode preserved, as implemented by `digest`). The primary bootstrap uses 10,000 resamples, seed `2026101303`, and keeps each task's three repetitions together within task family. Original protocol identifiers are retained for consistency with the data.

The complete raw API archive is not included. Therefore this repository does not independently reproduce raw-response parsing, token usage, transport recovery or reconstruction of SC6 pools from original responses. SC6 mean differences can be checked from the supplied task-level data; its exploratory confidence intervals are retained reports. Full raw-response auditing requires the original archive. These boundaries also apply to the original analysis entry point, which requires the complete 24-shard archive.

The source includes optional live API execution, which incurs charges if explicitly used. Credentials are read from the local environment; no credentials or local configuration files are included. The documented offline command does not invoke the provider. The live dependency ranges are not an exact historical environment lock. No new paid experiment was run to prepare this export.

## License

Source code, verification scripts, documentation, and the HTML catalogue's interface code are licensed under the [MIT License](LICENSE). The datasets and result records in `evidence/`, and the question/answer content displayed in the catalogue, are licensed under [CC BY 4.0](LICENSE-DATA.md). Reuse of the research data requires appropriate attribution, a license link, and an indication of any changes.
