import json
import os
import tempfile
from dataclasses import dataclass, field
from tqdm import tqdm

from kanha.utils.logging import get_logger
from kanha.lam.environment import TaskEnvironment
from kanha.lam.agent import TaskAgent

logger = get_logger(__name__)

@dataclass
class EvalReport:
    total_goals: int = 0
    successes: int = 0
    failures: int = 0
    aborts: int = 0
    overall_success_rate: float = 0.0
    per_tier_success: dict = field(default_factory=dict)
    per_split_success: dict = field(default_factory=dict)
    syntax_valid_rate: float = 0.0
    avg_steps: float = 0.0
    avg_optimal_steps: float = 0.0
    avg_steps_ratio: float = 0.0
    loop_hit_rate: float = 0.0
    step_limit_hit_rate: float = 0.0
    wrong_side_effect_rate: float = 0.0
    details: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "LAM Evaluation Report",
            "=" * 50,
            f"Total Goals:             {self.total_goals}",
            f"Success Rate:            {self.overall_success_rate:.2%}",
            f"  Successes:             {self.successes}",
            f"  Failures:              {self.failures}",
            f"  Aborts:                {self.aborts}",
            f"Loop Hit Rate:           {self.loop_hit_rate:.2%}",
            f"Step Limit Hit Rate:     {self.step_limit_hit_rate:.2%}",
            f"Wrong Side Effect Rate:  {self.wrong_side_effect_rate:.2%}",
            f"Avg Steps:               {self.avg_steps:.2f}",
            f"Avg Steps vs Optimal:    {self.avg_steps_ratio:.2f}x",
            f"Syntax Valid Rate:       {self.syntax_valid_rate:.2%}",
            "",
            "Per-Tier Breakdown:",
        ]
        for tier, data in sorted(self.per_tier_success.items()):
            lines.append(f"  {tier}: {data['success']}/{data['total']} = {data['rate']:.2%}")
        if self.per_split_success:
            lines.append("")
            lines.append("Per-Split Breakdown:")
            for split, data in sorted(self.per_split_success.items()):
                lines.append(f"  {split}: {data['success']}/{data['total']} = {data['rate']:.2%}")
        return "\n".join(lines)

class LAMEvaluator:
    def __init__(self, model, tokenizer, test_data_path: str, reference_date=None):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data_path = test_data_path
        self.reference_date = reference_date
        
        self.test_cases = []
        with open(test_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.test_cases.append(json.loads(line))
                    
    def evaluate(self) -> EvalReport:
        report = EvalReport()
        report.total_goals = len(self.test_cases)
        
        if report.total_goals == 0:
            return report
        
        # Use a temp directory for ephemeral task stores
        tmpdir = tempfile.mkdtemp()
        tmp_store_path = os.path.join(tmpdir, "eval_tasks.json")
        
        env = TaskEnvironment(task_store_path=tmp_store_path, reference_date=self.reference_date)
        agent = TaskAgent(self.model, self.tokenizer, env)

        
        loop_hits = 0
        limit_hits = 0
        wrong_side_effects = 0
        total_steps = 0
        total_optimal = 0
        syntax_valid = 0
        total_actions = 0
        
        for case in tqdm(self.test_cases, desc="Evaluating LAM"):
            goal = case.get('goal', '')
            world = case.get('world', [])
            expected_state = case.get('expected_state', [])
            expected_reply = case.get('expected_reply', None)
            tier = case.get('tier', 'unknown')
            split = case.get('split', 'unknown')
            opt_steps = case.get('optimal_steps', 1)
            
            # Reset environment to initial world state
            env.reset(initial_tasks=world)
            initial_state = env.get_state()
            
            result = agent.run(goal)
            
            final_state = env.get_state()
            
            # snapshot() returns a dict keyed by ID; extract the task dicts
            if isinstance(final_state, dict):
                final_tasks = list(final_state.values())
            else:
                final_tasks = final_state
            
            # expected_state may also be a dict or list
            if isinstance(expected_state, dict):
                expected_tasks = list(expected_state.values())
            else:
                expected_tasks = expected_state
            
            state_match = self._compare_states(expected_tasks, final_tasks)
            
            reply_match = True
            if expected_reply:
                reply_match = self._check_reply(goal, result.final_message or "", expected_reply)
                
            side_effect = self._check_side_effects(initial_state, final_tasks, expected_tasks)

            if side_effect:
                wrong_side_effects += 1
                
            success = state_match and reply_match and not side_effect
            
            if result.aborted:
                report.aborts += 1
                if result.abort_reason == 'action loop detected':
                    loop_hits += 1
                elif result.abort_reason == 'step limit reached':
                    limit_hits += 1
                    
            if success:
                report.successes += 1
            else:
                report.failures += 1
                
            total_steps += result.num_steps
            total_optimal += opt_steps
            
            for step in result.steps:
                total_actions += 1
                if step.get('valid', False):
                    syntax_valid += 1
                    
            if tier not in report.per_tier_success:
                report.per_tier_success[tier] = {'total': 0, 'success': 0, 'rate': 0.0}
            report.per_tier_success[tier]['total'] += 1
            if success:
                report.per_tier_success[tier]['success'] += 1
                
            if split not in report.per_split_success:
                report.per_split_success[split] = {'total': 0, 'success': 0, 'rate': 0.0}
            report.per_split_success[split]['total'] += 1
            if success:
                report.per_split_success[split]['success'] += 1
                
            report.details.append({
                'goal': goal,
                'success': success,
                'aborted': result.aborted,
                'reason': result.abort_reason,
                'steps': result.num_steps
            })
            
        report.overall_success_rate = report.successes / report.total_goals
        report.loop_hit_rate = loop_hits / report.total_goals
        report.step_limit_hit_rate = limit_hits / report.total_goals
        report.wrong_side_effect_rate = wrong_side_effects / report.total_goals
        report.avg_steps = total_steps / report.total_goals
        report.avg_optimal_steps = total_optimal / report.total_goals
        report.avg_steps_ratio = report.avg_steps / max(1.0, report.avg_optimal_steps)
        if total_actions > 0:
            report.syntax_valid_rate = syntax_valid / total_actions
            
        for t in report.per_tier_success:
            report.per_tier_success[t]['rate'] = report.per_tier_success[t]['success'] / max(1, report.per_tier_success[t]['total'])
        for s in report.per_split_success:
            report.per_split_success[s]['rate'] = report.per_split_success[s]['success'] / max(1, report.per_split_success[s]['total'])
            
        return report
        
    def _compare_states(self, expected: list, actual: list) -> bool:
        def canonicalize(tasks):
            return sorted([
                (
                    t.get('title'),
                    t.get('due'),
                    t.get('priority', t.get('prio', 'med')),
                    t.get('tag'),
                    t.get('status', 'todo'),
                )
                for t in tasks
            ])
        return canonicalize(expected) == canonicalize(actual)
        
    def _check_reply(self, goal: str, reply: str, expected: dict) -> bool:
        if not reply:
            return False
        reply_lower = reply.lower()
        if 'contains_number' in expected:
            if str(expected['contains_number']) not in reply_lower:
                return False
        if 'contains_title' in expected:
            if str(expected['contains_title']).lower() not in reply_lower:
                return False
        return True
        
    def _check_side_effects(self, initial_state, final_state: list, expected_state: list) -> bool:
        """Returns True if there are UNEXPECTED state changes."""
        def get_task_tuples(tasks):
            if isinstance(tasks, dict):
                tasks = list(tasks.values())
            return set(
                (
                    t.get('title'),
                    t.get('due'),
                    t.get('priority', t.get('prio', 'med')),
                    t.get('tag'),
                    t.get('status', 'todo'),
                )
                for t in tasks
            )
            
        final_set = get_task_tuples(final_state)
        expected_set = get_task_tuples(expected_state)
        
        # Any differences (tasks in final but not in expected) are wrong side effects
        diff = final_set - expected_set
        return len(diff) > 0


