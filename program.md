# autoresearch — Negotiation Challenge

This is an autonomous research loop for maximizing score in the following negotiation game.

Write a strategy prompt that guides an AI agent through multi-round resource negotiations. Your prompt is injected into the system instructions of a Gemini model that negotiates on your behalf against a baseline opponent. The goal: maximize your score across 10 games.

You are autonomous. Think like a professor of economics and game theory. If you run out of ideas, think harder — consider game theory, negotiation literature, information asymmetry, signaling theory. Try combining near-misses from prior experiments. Try more radical approaches (deceptive anchoring, cooperative information-sharing, mixed strategies). The loop runs until the human interrupts you, period.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr9`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
   **Prerequisite** — ensure `GEMINI_API_KEY` is exported in your shell before running any `uv` commands:
   ```bash
   export GEMINI_API_KEY=<your-api-key>
   ```
3. **Read the in-scope files**:
   - `README.md` — rules, CLI, scoring overview.
   - `src/negotiate/engine.py` — game constants, scenario generation, prompt building. **Do not modify.**
   - `src/negotiate/runner.py` — orchestration logic. **Do not modify.**
   - `prompts/cooperative.txt` and `prompts/aggressive.txt` — baseline strategy examples.
4. **Create your strategy file**: `prompts/autoresearch.txt` (max 2000 characters). This is the **only file you edit**. After each commit, the prompt is archived as `prompts/autoresearch_{run_id}.txt` (untracked, where `{run_id}` is an integer starting from 1 and incrementing with each experiment) so that every prior strategy is available as context for future experiments.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good, then start the loop.

## The Game (summary)

Two AI agents negotiate to split a pool of resources (books, hats, balls). Each player has **private valuations** — you can't see what the other player's resources are worth to them.

- **5 rounds** per game. Player A moves first each round.
- **Actions**: `propose` a split, `accept` the last proposal, or `reject` it. Every action includes a public message visible to the opponent.
- **Shared history**: Both players see the full negotiation history — all proposals, actions, and messages from every prior turn.
- **Scoring**: If a deal is reached, your score is `sum(your_valuation * quantity_you_receive) / max_possible` (range 0.0–1.0). No deal after 5 rounds = **-0.5** for both players.
- **Role alternation**: Half your games you play as Player A (move first), half as Player B (move second).
- **10 games** per evaluation, each with a unique scenario. Your final score is the mean across all games.

**The goal is simple: maximize the mean score across 10 games.**

## What you CAN do

- Edit `prompts/autoresearch.txt` — the strategy prompt injected into the AI agent's system instructions under `ADDITIONAL STRATEGY INSTRUCTIONS`. Max 2000 characters.
- Read `prompts/autoresearch_*.txt` — the accumulated archive of all previous experiment prompts (named `autoresearch_1.txt`, `autoresearch_2.txt`, etc.). Use these as context when designing new strategies to avoid repeating failures and to build on successes.
- Explore different negotiation philosophies: cooperative, competitive, adaptive, Bayesian, etc.
- Use messaging strategically — messages are public and visible to the opponent.

## What you CANNOT do

- Modify any `.py` files — the engine, runner, inference, and CLI are read-only.
- Install new packages or add dependencies.
- Exceed 2000 characters in the strategy prompt (the harness silently truncates).

## Output format

Running an experiment prints a summary per game and a final aggregate like:

```
Game 0: score=0.823 (deal in round 2)
Game 1: score=-0.500 (no deal)
...
Mean score: 0.612
```

Extract the key metric:

```bash
export GEMINI_API_KEY=<your-api-key>  # must be set before running
for seed in 42 99 7; do
  uv run negotiate test prompts/autoresearch.txt -n 10 -s $seed > results/run_{run_id}_s${seed}.log 2>&1
done
grep "Mean score" results/run_{run_id}_s*.log
```

The **mean of the per-seed mean scores** is the decision metric. This prevents overfitting to a single seed.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated — commas break in descriptions).

The TSV has a header row and 7 columns:

```
commit	mean_score	score_s42	score_s99	score_s7	no_deal_rate	status	description
```

1. git commit hash (short, 7 chars)
2. mean score across all seeds (average of columns 3–5) — use -0.500000 for crashes
3. mean score for seed 42
4. mean score for seed 99
5. mean score for seed 7
6. no-deal rate as a fraction averaged across seeds (e.g. 0.10) — use 1.0 for crashes
7. status: `keep`, `discard`, or `crash`
8. short text description of what this experiment tried

Example:

```
commit	mean_score	score_s42	score_s99	score_s7	no_deal_rate	status	description
a1b2c3d	0.612345	0.623000	0.601000	0.613035	0.10	keep	baseline cooperative
b2c3d4e	0.651200	0.660000	0.645000	0.648600	0.00	keep	add deadline pressure + information probing
c3d4e5f	0.590000	0.610000	0.580000	0.580000	0.20	discard	aggressive opening bid
d4e5f6g	-0.500000	-0.500000	-0.500000	-0.500000	1.0	crash	malformed strategy caused parse error
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/apr9`).

LOOP FOREVER:

1. Look at the git state: current branch and commit. Determine the next `{run_id}` by counting existing `prompts/autoresearch_*.txt` files (e.g. `ls prompts/autoresearch_*.txt 2>/dev/null | wc -l`) and adding 1. Read all previous archived prompts for context on what strategies have been tried and how they performed.
2. Edit `prompts/autoresearch.txt` with a new strategy idea, informed by the archive of previous prompts.
3. `git add prompts/autoresearch.txt && git commit -m "experiment: <short description>"`
4. Archive the prompt: `cp prompts/autoresearch.txt prompts/autoresearch_{run_id}.txt` (leave untracked).
5. Run the experiment across 3 seeds (ensure `GEMINI_API_KEY` is exported):
   ```
   for seed in 42 99 7; do
     uv run negotiate test prompts/autoresearch.txt -n 10 -s $seed > results/run_{run_id}_s${seed}.log 2>&1
   done
   ```
6. Read the results from all seeds: `grep "Mean score" results/run_{run_id}_s*.log`
7. Compute the **mean of the 3 per-seed scores** — this is the decision metric.
8. If any seed's output is empty or malformed, run `tail -n 50 results/run_{run_id}_s${seed}.log` to diagnose. Attempt a fix. Give up after a few retries.
9. Record the results in `results.tsv` (do **not** commit this file — leave it untracked). Log individual seed scores in the per-seed columns.
10. If the multi-seed mean score improved (higher), **advance**: keep the commit.
11. If the multi-seed mean score is equal or worse, `git reset --hard HEAD~1` and discard. The archived `prompts/autoresearch_{run_id}.txt` file remains as a record of what was tried.

A strategy that scores well on one seed but poorly on others is overfitting — discard it.

A strategy that scores well on one seed but poorly on others is overfitting — discard it.

## Writing a strategy

Your strategy prompt is appended to the system instructions under `ADDITIONAL STRATEGY INSTRUCTIONS`. The model already knows the game rules, resource pool, and its own valuations — your job is to tell it **how to negotiate**.

Things to consider:

- **Information asymmetry**: You know your valuations but not your opponent's. Resources you value little might be gold to them — and vice versa.
- **No-deal penalty**: -0.5 is the worst outcome. Any deal that gives you even 1 point beats no deal.
- **Role awareness**: You play as both A (first mover) and B (second mover). Your strategy should work from either position.
- **Round pressure**: The final round warning creates urgency — should you hold firm or compromise?

Example strategies are in `prompts/`.

## Research methodology

Do not explore randomly. Follow a structured, hypothesis-driven approach organized in phases.

### Phase 1: Broad exploration

Systematically test one idea from each **strategy family** below. Run each in isolation to build a map of what works. The goal is to cover the search space quickly.

### Phase 2: Refinement

Take the top 2–3 performing strategies. Try combinations, parameter tweaks, and messaging variations. Iterate on what worked.

### Phase 3: Ablation

When a strategy scores well, remove components one at a time to find the minimal effective prompt. Simpler is better — strip anything that doesn't contribute.

### Phase 4: Robustness

Since every experiment already runs 3 seeds, focus this phase on stress-testing with additional seeds (e.g. `-s 123`, `-s 256`) and confirming the top strategy generalizes beyond the standard set.

## Strategy taxonomy

Use this as a checklist of game-theoretic approaches. Track which families you've explored in `results.tsv` descriptions.

### Opening & anchoring
- **Aggressive anchor**: Open with a lopsided first offer to shift the bargaining range.
- **Fair-split opener**: Open with 50/50 to signal cooperation and speed up agreement.
- **Exploratory opener**: Make a probe offer designed to reveal opponent valuations rather than close a deal.

### Information extraction
- **Preference probing**: Use messages to ask about opponent preferences. Offer trades on your low-value items to infer their valuations.
- **Revealed preference**: Observe which items opponents fight hardest for — those are their high-value items.
- **Strategic signaling**: Misrepresent your own preferences to gain advantage (claim to value items you don't).

### Concession dynamics
- **Tit-for-tat concessions**: Match opponent concessions proportionally. Rewarding cooperation, punishing stubbornness.
- **Asymmetric concessions**: Concede heavily on items you value little (which may be valuable to them), hold firm on your high-value items.
- **Deadline pressure**: Concede slowly in early rounds, signal urgency in final rounds to force acceptance.
- **Boulwarism**: Make a near-final offer early and barely move. High-risk but can capture surplus if opponent is eager to deal.

### Cooperative / integrative
- **Logrolling**: Explicitly propose trading low-value items for high-value items. Seek win-win splits.
- **Value creation framing**: Use messages to frame proposals as mutually beneficial. "I'll give you all the X if I can have the Y."
- **Package deals**: Bundle items together rather than negotiating item-by-item.

### Competitive / distributive
- **Good cop / bad cop**: Alternate between aggressive and cooperative messaging tones across rounds.
- **Commitment tactics**: Use messages to credibly commit to a position ("This is my final offer").
- **Strategic rejection**: Reject offers that are acceptable but below your aspiration level, gambling on better terms.

### Adaptive / meta
- **Bayesian updating**: Update beliefs about opponent valuations based on their proposals and messages. Adjust strategy accordingly.
- **Round-conditional behavior**: Different strategy for rounds 1-2 (explore) vs 3-4 (negotiate) vs 5 (close).
- **Role-conditional behavior**: Different opening strategy as Player A (first mover advantage) vs Player B (respond and counter).
- **Opponent modeling**: Detect if the opponent is cooperative or competitive from early moves and adapt.

### Risk management
- **No-deal avoidance**: In final round, accept any offer that gives positive value. -0.5 is catastrophic.
- **Satisficing**: Set a target score threshold; accept any deal above it rather than holding out for perfection.
- **Mixed strategies**: Randomize between approaches to be less predictable (though within a single game, consistency may matter more).

## Analysis discipline

After each experiment, don't just record the mean score. Analyze the per-game results:

1. **Read all game lines**: `grep "Game [0-9]" results/run_{run_id}_s*.log` — look at the score distribution across all seeds, not just the mean.
2. **Identify no-deal games**: Which games ended without a deal? Is the pattern consistent across seeds or seed-specific?
3. **Identify low-scoring deals**: Which games had deals but poor scores? The strategy may have conceded too much.
4. **Check seed variance**: If results vary wildly between seeds, the strategy is brittle. Prefer strategies with tight score distributions across seeds.
5. **Compare against prior runs**: Did this strategy improve on specific games where the previous strategy was weak? Or did it regress on games that were already strong?
6. **Form the next hypothesis**: Based on the analysis, write a one-line hypothesis for the next experiment before editing the prompt. Record it in the commit message.

This analysis is what drives intelligent iteration. Without it, you're just rolling dice.

## Simplicity criterion

All else being equal, simpler strategy prompts are better. A complex prompt that adds 10 points of improvement is great. A complex prompt that adds 1 point is probably not worth the brittleness. If you can remove language and maintain the score — remove it.

## The first run

Your very first run should be to establish a baseline using the existing `prompts/cooperative.txt`:

```bash
export GEMINI_API_KEY=<your-api-key>  # must be set before running
for seed in 42 99 7; do
  uv run negotiate test prompts/cooperative.txt -n 10 -s $seed > results/run_baseline_s${seed}.log 2>&1
done
grep "Mean score" results/run_baseline_s*.log
```

Record this as the baseline in `results.tsv` with the current commit hash of the repo. Then create `prompts/autoresearch.txt` with your first experimental strategy and begin the loop.

## NEVER STOP

Once the experiment loop has begun (after initial setup), do **NOT** pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be away and expects you to continue working **indefinitely** until manually stopped.

You are autonomous. If you run out of ideas, go back to the **strategy taxonomy** — find families you haven't explored, try combining top performers, run ablations on your best prompt. Consult the analysis from prior runs to find weak spots. The loop runs until the human interrupts you, period.
