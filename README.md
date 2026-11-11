# Vision Large runner for ApprenticeBench

This Python runner connects Vispark Vision Large to an accounts-payable
evaluation environment through application tools. It enables native continual
learning, passes the latest state to the next Vision request, and writes a JSON
report from the environment's grades after the 100-task sequence.

**Official run status.** NeoCognition's public ApprenticeBench materials describe
the benchmark and show results and examples. This repository does not include
the official Odoo environment, ordered invoices, mentor simulation, or grader.
A runnable official integration requires those assets and an adapter for the
interface NeoCognition supplies. The current verification uses synthetic tasks
and mocked Vision responses. Contact `apprenticebench@neocognition.io` for
evaluation access.

[ApprenticeBench](https://apprenticebench.com/) ·
[Evaluation methodology](https://neocognition.io/blog/apprentice-bench) ·
[Vispark Lab](https://lab.vispark.in/)

## Verify locally

Use Python 3.9 or later. The runner uses the Python standard library.

```sh
python3 -m unittest discover -s tests -v
```

The tests cover the native request shape, multiple tool calls, tool-result
follow-up, released feedback, state carryover across bills, interrupted reports,
and a complete 100-task CLI run against a fixture. The fixture's score is
predetermined by the test grader.

## Run with an evaluation environment

Place the environment adapter in an importable Python module, for example
`private/apprentice_adapter.py`. From the repository directory, copy
`.env.example` to `.env` and set `VISION_API_KEY`. An exported
`VISION_API_KEY` takes precedence. Then run:

```sh
PYTHONPATH=private python3 benchmark.py \
  --environment apprentice_adapter:create_environment \
  --report runs/vision-large-cl.json
```

The command expects a fresh evaluation session and exactly 100 tasks. Set
`--max-calls-per-task` and, optionally, `--max-estimated-usd` to the run's
approved limits. The cost threshold is checked before each request against
reported token usage so far. The last request can take the estimate over the
threshold. The report includes each bill's pass/fail outcome, call count,
learning-state response count, runtime, and estimated token cost. Totals include
the provider's continual-learning usage and consumed units where returned.
The USD estimate applies $7.37 per million input tokens and $22.11 per million
output tokens. Any separately billed learning or cache activity needs to be
reconciled with provider billing.

The runner writes a partial report after each graded bill. An interrupted run
has no final success rate. The environment owns its session lifecycle and any
recovery procedure. Use a new session and report path for a new run.

## Environment adapter

`create_environment()` returns an object implementing `runner.Environment`.
This interface belongs to this repo. Adapt it to the authorized evaluator's
interface once access is available.

| Adapter member | Contract |
| --- | --- |
| `name`, `instructions` | An environment identifier and the instructions available to the apprentice. |
| `tools` | Vision function declarations for the permitted application tools. |
| `next_task()` | Return the next `runner.Task` or `None` when finished. Each task has a unique ID, native Vision content items, and optional feedback already released by the evaluator. |
| `call_tool(name, arguments)` | Invoke the application tool using the active evaluation session. Return a JSON-compatible value or a list of native Vision content items. Return recoverable tool failures as JSON so the model can correct its action. |
| `finish_task(task_id, final_text)` | Finalize the task through the evaluator and return its authoritative Boolean grade. The model does not see the grade unless the evaluator releases it as feedback. |

The adapter controls what the apprentice can read or change at each step.
Keep the historical workspace available as permitted by the benchmark and
expose only the records and feedback released at the current simulated date.
Bind application tools to the benchmark workspace. Validate tool arguments
in the adapter and avoid model access to the grader, future tasks, or a shell
running alongside the evaluator. The published benchmark has GUI and API
settings. This runner is designed for the API setting.

Vision requests use `size: "large"`, `stream: false`, and
`continual_learning: true`. The first request starts without a state blob.
The runner sends each subsequent request with the most recent nonempty
`data.weights` returned by Vision and stops if a task produces no state.
The native API guide describes feeding tool results back as follow-up requests
without specifying a separate tool-result field. This client sends the tool
call and its result as content in the current task's transcript. At the next
task it starts a fresh transcript with the new task and released feedback.
The state blob stays in memory and is omitted from reports. Confirm the
state round trip and tool follow-up with the live API before a scored run.
