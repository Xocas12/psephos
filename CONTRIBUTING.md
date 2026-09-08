# Contributing

psephos flags statistical patterns in precinct-level results. Its failure mode is not a crash;
it is a confident p-value handed to someone who already knew what they wanted to find. A weak
check merged into this repository launders a prior into a number, and the number carries this
repository's name with it. So the bar for a new check is written down here, and a check missing
any part of it will not be merged.

Read [INTERPRETATION.md](INTERPRETATION.md) before you write any code. It explains what the
output means and, more importantly, what it does not.

## Setup

Python 3.10 or newer.

```console
uv sync --all-extras
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
```

All three verification commands must pass before you propose a change. Continuous integration
runs these commands on every push and pull request, with the test suite repeated across Python
3.10 to 3.13, plus a guardrail job that constructs a flagged finding with no confounds and
expects the construction to raise.

## The bar for a new check

Six things. A check missing any of them will not be merged.
[src/psephos/methods/_template.py](src/psephos/methods/_template.py) is a working example of all
six against a deliberately trivial statistic, and `tests/test_template.py` is its test file.
Copy their shape before you invent your own.

### 1. A null stated in one sentence

Put the null at the top of the module docstring, where the existing checks put theirs, in one
sentence. If it cannot be stated in one sentence, it is not understood well enough to ship. A
p-value is only meaningful against a null the reader can see without reading your code.

### 2. A confounds list

Every flagged finding must name the innocent explanations that produce the same signal. This is
not a convention; it is enforced in code: `Finding.__post_init__` in
[src/psephos/_types.py](src/psephos/_types.py) raises if a finding flagged `notable` or `strong`
carries no confounds. Name the real confounds, not ceremonial ones. If the check touches a
percentage, one of them will be precinct size, and the confound should say how the check
controls for it.

### 3. A quiet test

The check must not fire on `clean_election()` from `tests/conftest.py`. Assert the flag is `ok`
and the p-value is above the significance level, not merely that nothing raised. This is the
test that matters most and the one most often skipped: a check that fires on honest data
launders suspicion, and it will not be merged however impressive the loud half looks. If the
check is size-sensitive, the quiet test should use the same exclusions the real check uses.

### 4. A loud test

The check must find a distortion injected by a named generator from `tests/conftest.py`, and the
measured effect must rise with the injected amount. Use the existing generators
(`with_rounded_turnout`, `with_stuffing`, `with_invented_last_digits`) rather than writing a new
one; if none of them injects the mechanism your check exists to find, say so in the pull
request, and add the generator to the conftest with the same care: it must be a named,
documented mechanism, not a fudge fitted to make your statistic move.

### 5. An underpowered path

When too few units survive the filters, the check must return a finding flagged `underpowered`
with no p-value, rather than a null that reads as a clean bill of health. `underpowered` is a
distinct outcome in this tool for exactly that reason; see [INTERPRETATION.md](INTERPRETATION.md).
Test the path: feed the check too few usable units and assert the flag, the missing p-value, and
the excluded count.

### 6. A citation, or an explicit statement of novelty

Anything taken from a paper carries the reference, in the module docstring, in the form the
existing checks use, including the standing caveat that psephos implements published
descriptions and has not been validated line by line against the authors' own code; issue #6
tracks that validation. Anything invented says so plainly and does not borrow authority. Do not
attach a citation you have not verified describes the statistic you actually wrote.

## What will not be merged

- A check without a quiet test.
- A first-digit Benford test, or any test the scale of the data cannot support. The reasoning is
  recorded in issue #8 and in INTERPRETATION.md.
- An estimator that converts a dependence into a count of anomalous votes, until its
  construction is validated against its source. Issue #9.
- A single composite score, or any change that gives a script a way to treat "flagged" as
  "failed". The decisions are recorded in ROADMAP.md.
- A check registered in `psephos.audit` before its six requirements exist and pass.

## House style

- ASCII only in source files. No em dashes anywhere.
- Docstrings carry the reasoning, not just the signature: what the null is, what the statistic
  cannot tell you, and which decisions were deliberate. Match the density of the modules you
  are editing.
- New behaviour arrives with a test that would fail without it.

## Where to start

Issues are organised by milestone in [ROADMAP.md](ROADMAP.md), and the ones labelled
`good first issue` on the issue tracker are scoped for a first contribution.
