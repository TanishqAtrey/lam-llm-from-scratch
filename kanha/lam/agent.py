from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from kanha.lam.actions import parse_action, validate_action, execute_action, ActionParseError, ACTION_SCHEMA
from kanha.lam.environment import TaskEnvironment
from kanha.lam.trajectory import format_step_prompt, GOAL_MARKER, ACTION_MARKER, OBSERVATION_MARKER
from kanha.core.generation import generate

@dataclass
class AgentResult:
    goal: str
    steps: list[dict] = field(default_factory=list)  # [{action, observation, valid, step_num}]
    final_message: Optional[str] = None
    success: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    num_steps: int = 0

class TaskAgent:
    def __init__(self, model, tokenizer, environment: TaskEnvironment, 
                 max_steps: int = 8, max_retries: int = 3):
        self.model = model
        self.tokenizer = tokenizer
        self.env = environment
        self.max_steps = max_steps
        self.max_retries = max_retries
    
    def run(self, goal: str) -> AgentResult:
        result = AgentResult(goal=goal)
        
        while result.num_steps < self.max_steps:
            history = self.env.get_trajectory()
            
            if self._detect_loop(history):
                result.aborted = True
                result.abort_reason = 'action loop detected'
                break
                
            prompt = format_step_prompt(goal, history)
            
            action_text = ""
            action_valid = False
            error_msg = ""
            
            for attempt in range(self.max_retries + 1):
                temp = 0.0 if attempt == 0 else 0.3
                raw_action = self._generate_action(prompt, temperature=temp)
                
                if not raw_action:
                    error_msg = "Empty action generated"
                    result.steps.append({
                        'action': '', 'observation': error_msg,
                        'valid': False, 'step_num': result.num_steps + 1
                    })
                    continue
                
                try:
                    name, args = parse_action(raw_action)
                    valid, msg = validate_action(name, args)
                    if not valid:
                        error_msg = f"Validation error: {msg}"
                        result.steps.append({
                            'action': raw_action, 'observation': error_msg,
                            'valid': False, 'step_num': result.num_steps + 1
                        })
                        continue
                    
                    action_text = raw_action
                    action_valid = True
                    break
                except ActionParseError as e:
                    error_msg = f"Parse error: {str(e)}"
                    result.steps.append({
                        'action': raw_action, 'observation': error_msg,
                        'valid': False, 'step_num': result.num_steps + 1
                    })
                    continue
            
            if not action_valid:
                result.aborted = True
                result.abort_reason = f"Action generation failed after retries: {error_msg}"
                break
            
            observation, is_terminal = self.env.step(action_text)
            
            step_record = {
                'action': action_text,
                'observation': observation,
                'valid': action_valid,
                'step_num': result.num_steps + 1
            }
            result.steps.append(step_record)
            result.num_steps += 1
            
            if is_terminal:
                result.final_message = observation
                result.success = True
                break
                
        if not result.success and not result.aborted:
            if result.num_steps >= self.max_steps:
                result.aborted = True
                result.abort_reason = 'step limit reached'
                
        return result
    
    def _generate_action(self, prompt: str, temperature: float = 0.0) -> str:
        # Generate with greedy/low temp
        raw = generate(self.model, self.tokenizer, prompt, max_new_tokens=64, temperature=temperature, top_k=1)
        return self._clean_action(raw)
    
    def _detect_loop(self, history: list[dict]) -> bool:
        """Returns True if the last 3 actions are identical."""
        if len(history) < 3:
            return False
        last_3 = history[-3:]
        # Ensure they all have 'action' keys
        for h in last_3:
            if 'action' not in h:
                return False
        return last_3[0]['action'] == last_3[1]['action'] == last_3[2]['action']
    
    def _clean_action(self, raw: str) -> str:
        """Cleans model output to extract just the action call."""
        clean = raw.strip()
        if not clean:
            return ""
            
        first_line = clean.split('\n')[0].strip()
        
        # Remove trailing markers if generated by model
        markers = [OBSERVATION_MARKER, ACTION_MARKER, GOAL_MARKER, '### Observation', '### Action', '### Goal']
        for marker in markers:
            if marker in first_line:
                first_line = first_line.split(marker)[0].strip()
                
        return first_line
