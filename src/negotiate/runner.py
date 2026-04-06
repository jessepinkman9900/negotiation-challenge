"""Game runner: orchestrates N negotiation games for a given prompt."""

import asyncio
import hashlib
import statistics
import time
from typing import Optional, Callable

from google import genai

from .engine import (
    MAX_ROUNDS,
    NO_DEAL_PENALTY,
    build_system_prompt,
    build_turn_prompt,
    generate_scenario,
    max_possible,
    score_split,
    validate_offer,
)
from .inference import call_gemini


async def run_game(
    client: genai.Client,
    scenario: dict,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Run one full negotiation game.

    Player A = user (with custom strategy), Player B = baseline (no custom strategy).
    """
    pool = scenario["pool"]
    vals_a = scenario["valuations_a"]
    vals_b = scenario["valuations_b"]

    sys_a = build_system_prompt("A", pool, vals_a, custom_strategy=user_prompt)
    sys_b = build_system_prompt("B", pool, vals_b)

    history: list[dict] = []
    deal_reached = False
    final_round = MAX_ROUNDS
    user_score = NO_DEAL_PENALTY
    baseline_score = NO_DEAL_PENALTY
    last_proposal: Optional[dict] = None
    last_proposer: Optional[str] = None

    for round_num in range(1, MAX_ROUNDS + 1):
        # -- Player A turn --
        turn_prompt_a = build_turn_prompt(history, "A", round_num, pool)
        result_a = await call_gemini(client, sys_a, turn_prompt_a, semaphore)

        if result_a is None:
            result_a = {
                "action": "reject",
                "message": "(no response)",
                "reasoning": "",
                "offer": None,
            }

        action_a = result_a["action"]
        offer_a = result_a.get("offer")

        if action_a == "accept":
            if last_proposer != "B" or last_proposal is None:
                action_a = "reject"
                result_a["action"] = "reject"

        if action_a == "propose":
            if offer_a is None or not validate_offer(offer_a, pool):
                action_a = "reject"
                result_a["action"] = "reject"
                offer_a = None
                result_a["offer"] = None

        turn_a = {
            "round": round_num,
            "player": "A",
            "action": action_a,
            "message": result_a["message"],
            "reasoning": result_a["reasoning"],
            "offer": offer_a,
        }
        history.append(turn_a)

        if action_a == "propose" and offer_a:
            last_proposal = offer_a
            last_proposer = "A"

        if action_a == "accept" and last_proposal is not None and last_proposer == "B":
            a_split = last_proposal["their_share"]
            b_split = last_proposal["my_share"]
            max_a = max_possible(vals_a, pool)
            max_b = max_possible(vals_b, pool)
            user_score = score_split(vals_a, a_split) / max_a if max_a > 0 else 0.0
            baseline_score = score_split(vals_b, b_split) / max_b if max_b > 0 else 0.0
            deal_reached = True
            final_round = round_num
            break

        # -- Player B turn --
        turn_prompt_b = build_turn_prompt(history, "B", round_num, pool)
        result_b = await call_gemini(client, sys_b, turn_prompt_b, semaphore)

        if result_b is None:
            result_b = {
                "action": "reject",
                "message": "(no response)",
                "reasoning": "",
                "offer": None,
            }

        action_b = result_b["action"]
        offer_b = result_b.get("offer")

        if action_b == "accept":
            if last_proposer != "A" or last_proposal is None:
                action_b = "reject"
                result_b["action"] = "reject"

        if action_b == "propose":
            if offer_b is None or not validate_offer(offer_b, pool):
                action_b = "reject"
                result_b["action"] = "reject"
                offer_b = None
                result_b["offer"] = None

        turn_b = {
            "round": round_num,
            "player": "B",
            "action": action_b,
            "message": result_b["message"],
            "reasoning": result_b["reasoning"],
            "offer": offer_b,
        }
        history.append(turn_b)

        if action_b == "propose" and offer_b:
            last_proposal = offer_b
            last_proposer = "B"

        if action_b == "accept" and last_proposal is not None and last_proposer == "A":
            a_split = last_proposal["my_share"]
            b_split = last_proposal["their_share"]
            max_a = max_possible(vals_a, pool)
            max_b = max_possible(vals_b, pool)
            user_score = score_split(vals_a, a_split) / max_a if max_a > 0 else 0.0
            baseline_score = score_split(vals_b, b_split) / max_b if max_b > 0 else 0.0
            deal_reached = True
            final_round = round_num
            break

    return {
        "scenario_seed": scenario["seed"],
        "deal_reached": deal_reached,
        "final_round": final_round,
        "user_score": round(user_score, 4),
        "baseline_score": round(baseline_score, 4),
        "turns": history,
        "pool": pool,
        "valuations_user": vals_a,
        "valuations_baseline": vals_b,
    }


async def run_evaluation(
    prompt: str,
    num_games: int = 10,
    seed: int | None = None,
    concurrency: int = 20,
    on_game_complete: Callable[[int, int, dict], None] | None = None,
) -> dict:
    """Run a full evaluation: N games, collect stats.

    Args:
        prompt: The strategy prompt text.
        num_games: Number of games to play.
        seed: Base seed for reproducibility. None = random.
        concurrency: Max concurrent Gemini API calls.
        on_game_complete: Callback(completed, total, game_result) for progress.

    Returns dict with keys: prompt, games, stats, elapsed.
    """
    client = genai.Client()
    semaphore = asyncio.Semaphore(concurrency)

    if seed is None:
        import random
        seed = random.randint(0, 2**31 - 1)

    def _seed_for(idx: int) -> int:
        return int(hashlib.sha256(f"{seed}:{idx}".encode()).hexdigest(), 16) % (2**31)

    scenarios = [generate_scenario(_seed_for(i)) for i in range(num_games)]

    tasks = [
        asyncio.ensure_future(run_game(client, scenario, prompt, semaphore))
        for scenario in scenarios
    ]

    results = []
    completed = 0
    t_start = time.monotonic()

    for coro in asyncio.as_completed(tasks):
        result = await coro
        completed += 1
        results.append(result)
        if on_game_complete:
            on_game_complete(completed, num_games, result)

    elapsed = time.monotonic() - t_start

    user_scores = [r["user_score"] for r in results]
    deals = sum(1 for r in results if r["deal_reached"])

    stats = {
        "mean": round(statistics.mean(user_scores), 4),
        "median": round(statistics.median(user_scores), 4),
        "min": round(min(user_scores), 4),
        "max": round(max(user_scores), 4),
        "std": round(statistics.stdev(user_scores), 4) if len(user_scores) > 1 else 0.0,
        "deals_reached": deals,
        "games_played": num_games,
        "deal_rate": round(deals / num_games, 2),
    }

    return {
        "prompt": prompt,
        "seed": seed,
        "games": results,
        "stats": stats,
        "elapsed": round(elapsed, 2),
    }
