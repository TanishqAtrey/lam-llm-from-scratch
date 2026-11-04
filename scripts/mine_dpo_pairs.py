"""
scripts/mine_dpo_pairs.py
Automatic DPO Negative Mining from Failed Trajectories.

Runs the LAM agent on the evaluation dataset and captures:
  - Successful trajectories → chosen (positive) examples
  - Failed trajectories     → rejected (negative) examples

Pairs them by matching goals to create (chosen, rejected) DPO
preference pairs. Saves to data/lam/dpo_pairs.jsonl.

This makes the DPO pipeline functional — without this, dpo_train.py
had the infrastructure but no data, making it dead code.

The self-improving loop:
  1. Train LAM-SFT → agent
  2. Run agent on eval set → collect successes + failures
  3. Mine DPO pairs from the runs
  4. Train LAM-DPO with the pairs → better agent
  5. Repeat

Run:
    python main.py mine-dpo \
        --model models/lam/lam_final.pt \
        --eval_data data/lam/test.jsonl \
        --output data/lam/dpo_pairs.jsonl \
        --max_goals 500
"""

import json
import os
import tempfile
from dataclasses import dataclass, field
from collections import defaultdict

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.lam.environment import TaskEnvironment
from kanha.lam.agent import TaskAgent
from kanha.lam.task_store import TaskStore
from kanha.lam.trajectory import format_step_prompt
from kanha.utils.helpers import ensure_dir, get_device
from kanha.utils.logging import get_logger

log = get_logger("mine_dpo")


def _canonical_task(t: dict, reference_date: str) -> tuple:
    due = t.get('due')
    if due:
        due = TaskStore.resolve_symbolic_date(due, reference_date)
    return (
        t.get('title') or '',
        due or '',
        t.get('priority', t.get('prio', 'med')) or 'med',
        t.get('tag') or '',
        t.get('status', 'todo') or 'todo',
    )


def _is_true_success(case: dict, result, env: TaskEnvironment, reference_date: str) -> bool:
    """Verifies that the trajectory actually achieved true task success."""
    if result.aborted:
        return False

    final_state = env.get_state()
    final_tasks = list(final_state.values()) if isinstance(final_state, dict) else final_state
    expected_state = case.get('expected_state', [])
    expected_tasks = list(expected_state.values()) if isinstance(expected_state, dict) else expected_state

    def canonicalize(tasks):
        return sorted([_canonical_task(t, reference_date) for t in tasks])

    if canonicalize(expected_tasks) != canonicalize(final_tasks):
        return False

    # Check side effects
    initial_tasks = case.get('world', [])
    init_set = set(_canonical_task(t, reference_date) for t in (list(initial_tasks.values()) if isinstance(initial_tasks, dict) else initial_tasks))
    final_set = set(_canonical_task(t, reference_date) for t in final_tasks)
    exp_set = set(_canonical_task(t, reference_date) for t in expected_tasks)
    untouched_expected = init_set & exp_set
    if (untouched_expected - final_set) or (final_set - (init_set | exp_set)):
        return False

    expected_reply = case.get('expected_reply')
    if expected_reply:
        reply = (result.final_message or "").lower()
        if 'contains_number' in expected_reply and str(expected_reply['contains_number']) not in reply:
            return False
        if 'contains_title' in expected_reply and str(expected_reply['contains_title']).lower() not in reply:
            return False
        if 'contains_phrase' in expected_reply and str(expected_reply['contains_phrase']).lower() not in reply:
            return False

    return True


@dataclass
class MinedPair:
    """A single (chosen, rejected) DPO pair for one goal."""
    goal: str
    prompt: str        # the LAM prompt (goal + any shared prefix)
    chosen: str        # action from the successful trajectory
    rejected: str      # action from the failed trajectory
    tier: str = ""
    chosen_confidence: float = 0.0
    rejected_confidence: float = 0.0


@dataclass
class MiningReport:
    """Summary statistics from the mining run."""
    total_goals: int = 0
    goals_with_success: int = 0
    goals_with_failure: int = 0
    goals_with_both: int = 0    # goals where we got both → can mine pairs
    pairs_mined: int = 0
    oracle_fallback_pairs: int = 0  # pairs using oracle as chosen
    avg_chosen_confidence: float = 0.0
    avg_rejected_confidence: float = 0.0

    def summary(self) -> str:
        return (
            f"DPO Mining Report\n"
            f"{'='*40}\n"
            f"Total goals evaluated:  {self.total_goals}\n"
            f"Goals with success:     {self.goals_with_success}\n"
            f"Goals with failure:     {self.goals_with_failure}\n"
            f"Goals with both:        {self.goals_with_both}\n"
            f"Pairs mined:            {self.pairs_mined}\n"
            f"  From agent runs:      {self.pairs_mined - self.oracle_fallback_pairs}\n"
            f"  Oracle fallback:      {self.oracle_fallback_pairs}\n"
            f"Avg chosen confidence:  {self.avg_chosen_confidence:.2%}\n"
            f"Avg rejected confidence:{self.avg_rejected_confidence:.2%}\n"
        )


def _extract_first_action_with_prompt(goal: str, steps: list[dict]) -> tuple[str, str]:
    """
    Extracts the prompt and first action from a trajectory.
    Returns (prompt, action) — used for DPO pairing.
    """
    prompt = format_step_prompt(goal, [])  # prompt for the FIRST action
    if steps:
        action = steps[0].get("action", "")
    else:
        action = ""
    return prompt, action


def _find_divergence_pair(
    goal: str,
    chosen_steps: list[dict],
    rejected_steps: list[dict],
    chosen_confs: list[float] = None,
    rejected_confs: list[float] = None,
    retry_history: list[dict] = None,
) -> tuple[str, str, str, float, float] | None:
    """
    Finds the first step where chosen and rejected trajectories diverge.
    Checks all steps 0..N, creates the prompt up to that step using
    format_step_prompt(goal, steps[:step_idx]), and returns
    (prompt, chosen_action, rejected_action, chosen_conf, rejected_conf).
    """
    chosen_confs = chosen_confs or []
    rejected_confs = rejected_confs or []
    retry_history = retry_history or []

    min_len = min(len(chosen_steps), len(rejected_steps))
    for i in range(min_len):
        c_act = chosen_steps[i].get("action", "").strip()
        r_act = rejected_steps[i].get("action", "").strip()

        # If rejected step has empty action, check if retry_history captured it
        if not r_act and retry_history:
            for r in retry_history:
                if (r.get("step_num") == i + 1 or len(retry_history) == 1) and r.get("action"):
                    r_act = r.get("action", "").strip()
                    break

        # Fallback: if rejected generated empty action / immediate abort
        if not r_act and not rejected_steps[i].get("valid", True):
            r_act = "\n"

        if c_act and (r_act or r_act == "\n") and c_act != r_act:
            prompt = format_step_prompt(goal, chosen_steps[:i])
            c_conf = chosen_confs[i] if i < len(chosen_confs) else 1.0
            r_conf = rejected_confs[i] if i < len(rejected_confs) else 0.0
            return prompt, c_act, r_act, c_conf, r_conf

    # Check if rejected failed at the next step after min_len matching steps
    if len(chosen_steps) > min_len:
        i = min_len
        c_act = chosen_steps[i].get("action", "").strip()
        r_act = ""
        r_conf = 0.0
        for r in retry_history:
            if (r.get("step_num") == i + 1 or len(retry_history) == 1) and r.get("action"):
                r_act = r.get("action", "").strip()
                r_conf = r.get("confidence", 0.0)
                break
        if not r_act and retry_history:
            r_act = retry_history[-1].get("action", "").strip()
            r_conf = retry_history[-1].get("confidence", 0.0)

        # Fallback: if rejected halted prematurely or generated empty output
        if not r_act:
            r_act = "\n"

        if c_act and (r_act or r_act == "\n") and c_act != r_act:
            prompt = format_step_prompt(goal, chosen_steps[:i])
            c_conf = chosen_confs[i] if i < len(chosen_confs) else 1.0
            return prompt, c_act, r_act, c_conf, r_conf

    return None


def mine_dpo_pairs(args):
    """
    Main DPO mining function. Called from main.py mine-dpo command.

    Strategy:
      1. For each eval goal, run the agent and record the trajectory.
      2. Also compute the oracle solution for each goal.
      3. Build DPO pairs:
         a) If agent succeeded AND we have a failed run → pair them
         b) If agent failed AND oracle exists → oracle = chosen, agent = rejected
         c) If agent succeeded but no failure → skip (no negative)

    This gives us pairs where the model can learn "what to do" vs "what NOT to do".
    """
    device = get_device()

    # ── Load model ──
    log.info(f"Loading model from {args.model}")
    model = KanhaModel.from_pretrained(args.model)
    model.eval()
    model.to(device)
    tokenizer = KanhaTokenizer()

    # ── Load eval data ──
    eval_cases = []
    with open(args.eval_data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                eval_cases.append(json.loads(line))

    max_goals = getattr(args, "max_goals", len(eval_cases))
    eval_cases = eval_cases[:max_goals]

    log.info(f"Mining DPO pairs from {len(eval_cases)} goals")

    # ── Run agent on each goal (twice with different temps for diversity) ──
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_store = os.path.join(tmpdir, "mine_tasks.json")
        reference_date = getattr(args, "reference_date", "2025-01-15")

        env = TaskEnvironment(task_store_path=tmp_store, reference_date=reference_date)
        agent = TaskAgent(model, tokenizer, env)

        # Collect results per goal
        successes = {}   # goal_idx → AgentResult
        failures = {}    # goal_idx → AgentResult
        oracle_steps_by_goal = {}  # goal_idx → oracle steps list

        report = MiningReport(total_goals=len(eval_cases))

        for idx, case in enumerate(eval_cases):
            goal = case.get("goal", "")
            world = case.get("world", [])
            oracle_steps = case.get("steps", [])
            tier = case.get("tier", "unknown")

            if not goal:
                continue

            # Store oracle steps if available
            if oracle_steps:
                oracle_steps_by_goal[idx] = oracle_steps

            # Run 1: deterministic (temp=0 via agent default)
            env.reset(initial_tasks=world)
            result = agent.run(goal)

            if _is_true_success(case, result, env, reference_date):
                successes[idx] = result
            else:
                failures[idx] = result

            # Run 2: with slight temperature and relaxed top_k for diverse failures
            env.reset(initial_tasks=world)
            old_threshold = agent.CONFIDENCE_THRESHOLD
            old_top_k = agent._gen_top_k
            agent.CONFIDENCE_THRESHOLD = 0.0  # don't gate on confidence for mining
            agent._gen_top_k = 50             # relax top_k so temp=0.3 generates diverse trajectories
            result2 = agent.run(goal, temperature=0.3)
            agent.CONFIDENCE_THRESHOLD = old_threshold
            agent._gen_top_k = old_top_k

            if _is_true_success(case, result2, env, reference_date):
                if idx not in successes:
                    successes[idx] = result2
            else:
                if idx not in failures:
                    failures[idx] = result2

            if (idx + 1) % 50 == 0:
                log.info(f"Processed {idx+1}/{len(eval_cases)} goals "
                         f"({len(successes)} successes, {len(failures)} failures)")

    report.goals_with_success = len(successes)
    report.goals_with_failure = len(failures)

    # ── Build DPO pairs ──
    pairs: list[MinedPair] = []

    for idx, case in enumerate(eval_cases):
        goal = case.get("goal", "")
        tier = case.get("tier", "unknown")

        if not goal:
            continue

        # Case A: We have both success and failure for this goal
        if idx in successes and idx in failures:
            report.goals_with_both += 1

            success_result = successes[idx]
            failure_result = failures[idx]

            chosen_steps = [s for s in success_result.steps if s.get("valid")]
            rejected_steps = failure_result.steps if failure_result.steps else getattr(failure_result, "retry_history", [])

            divergence = _find_divergence_pair(
                goal=goal,
                chosen_steps=chosen_steps,
                rejected_steps=rejected_steps,
                chosen_confs=success_result.confidence_scores,
                rejected_confs=failure_result.confidence_scores,
                retry_history=getattr(failure_result, "retry_history", []),
            )

            if divergence:
                prompt, chosen_action, rejected_action, chosen_conf, rejected_conf = divergence
                pairs.append(MinedPair(
                    goal=goal, prompt=prompt,
                    chosen=chosen_action, rejected=rejected_action,
                    tier=tier,
                    chosen_confidence=chosen_conf,
                    rejected_confidence=rejected_conf,
                ))

        # Case B: Only failure — use oracle as chosen
        elif idx in failures and idx in oracle_steps_by_goal:
            failure_result = failures[idx]
            chosen_steps = oracle_steps_by_goal[idx]
            rejected_steps = failure_result.steps if failure_result.steps else getattr(failure_result, "retry_history", [])

            divergence = _find_divergence_pair(
                goal=goal,
                chosen_steps=chosen_steps,
                rejected_steps=rejected_steps,
                chosen_confs=[1.0] * len(chosen_steps),
                rejected_confs=failure_result.confidence_scores,
                retry_history=getattr(failure_result, "retry_history", []),
            )

            if divergence:
                prompt, chosen_action, rejected_action, chosen_conf, rejected_conf = divergence
                pairs.append(MinedPair(
                    goal=goal, prompt=prompt,
                    chosen=chosen_action, rejected=rejected_action,
                    tier=tier,
                    chosen_confidence=chosen_conf,
                    rejected_confidence=rejected_conf,
                ))
                report.oracle_fallback_pairs += 1

    report.pairs_mined = len(pairs)

    if pairs:
        chosen_confs = [p.chosen_confidence for p in pairs]
        rejected_confs = [p.rejected_confidence for p in pairs]
        report.avg_chosen_confidence = sum(chosen_confs) / len(chosen_confs)
        report.avg_rejected_confidence = sum(rejected_confs) / len(rejected_confs)

    # ── Save pairs ──
    output_path = args.output
    ensure_dir(os.path.dirname(output_path))

    with open(output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps({
                "prompt": pair.prompt,
                "chosen": pair.chosen,
                "rejected": pair.rejected,
                "goal": pair.goal,
                "tier": pair.tier,
                "chosen_confidence": pair.chosen_confidence,
                "rejected_confidence": pair.rejected_confidence,
            }, ensure_ascii=False) + "\n")

    log.info(f"\n{report.summary()}")
    log.info(f"DPO pairs saved to {output_path}")

    return report
