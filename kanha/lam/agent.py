from __future__ import annotations
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from kanha.lam.actions import parse_action, validate_action, execute_action, ActionParseError, ACTION_SCHEMA
from kanha.lam.environment import TaskEnvironment
from kanha.lam.trajectory import format_step_prompt, GOAL_MARKER, ACTION_MARKER, OBSERVATION_MARKER
from kanha.core.generation import generate, generate_with_confidence
from kanha.utils.config import cfg

@dataclass
class AgentResult:
    goal: str
    steps: list[dict] = field(default_factory=list)  # [{action, observation, valid, step_num, confidence}]
    final_message: Optional[str] = None
    success: bool = False
    aborted: bool = False
    abort_reason: Optional[str] = None
    num_steps: int = 0
    confidence_scores: list[float] = field(default_factory=list)  # per-action mean confidence
    low_confidence_count: int = 0  # how many actions were below threshold
    retry_history: list[dict] = field(default_factory=list)  # retry drafts and failures

class TaskAgent:
    """
    LAM agent that executes goal-directed action sequences.

    Now includes confidence scoring: each generated action is assigned
    a mean token-level probability. If confidence is below the threshold,
    the agent asks for clarification instead of blindly executing.
    """

    def __init__(self, model, tokenizer, environment: TaskEnvironment,
                 max_steps: int = None, max_retries: int = None,
                 confidence_threshold: float = None):
        self.model = model
        self.tokenizer = tokenizer
        self.env = environment

        # Read defaults from config, fall back to sensible values
        lam_cfg = cfg.lam
        self.max_steps = max_steps if max_steps is not None else getattr(lam_cfg, 'max_steps_per_trajectory', 8)
        self.max_retries = max_retries if max_retries is not None else getattr(lam_cfg, 'max_retries_on_invalid', 3)

        # Confidence thresholds from config
        self.CONFIDENCE_THRESHOLD = (
            confidence_threshold if confidence_threshold is not None
            else getattr(lam_cfg, 'confidence_threshold', 0.3)
        )
        self.CONFIDENCE_WARNING = getattr(lam_cfg, 'confidence_warning', 0.5)

        # Generation params from config
        gen_cfg = getattr(lam_cfg, 'generation', None)
        self._gen_max_tokens = getattr(gen_cfg, 'max_new_tokens', 64) if gen_cfg else 64
        self._gen_top_k = getattr(gen_cfg, 'top_k', 1) if gen_cfg else 1
    
    def run(self, goal: str, temperature: Optional[float] = None) -> AgentResult:
        result = AgentResult(goal=goal)
        # Clear stale history from prior runs so prompts aren't polluted
        self.env._history = []
        
        while result.num_steps < self.max_steps:
            history = self.env.get_trajectory()
            
            if self._detect_loop(history):
                result.aborted = True
                result.abort_reason = 'action loop detected'
                break
                
            prompt = format_step_prompt(goal, history)
            
            action_text = ""
            action_valid = False
            action_confidence = 0.0
            error_msg = ""
            last_failed_action = ""
            
            for attempt in range(self.max_retries + 1):
                if attempt == 0:
                    temp = temperature if temperature is not None else 0.0
                    k = self._gen_top_k
                else:
                    temp = max(temperature if temperature is not None else 0.0, 0.3)
                    # Relax top_k on retry so non-zero temperature produces diverse alternative actions
                    k = max(self._gen_top_k, 50)
                raw_action, confidence = self._generate_action(prompt, temperature=temp, top_k=k)
                
                if not raw_action:
                    error_msg = "Empty action generated"
                    last_failed_action = ""
                    result.retry_history.append({
                        'action': '', 'observation': error_msg,
                        'valid': False, 'step_num': result.num_steps + 1,
                        'confidence': 0.0,
                    })
                    continue

                # ── Confidence gate: refuse if model is too uncertain ──
                if confidence < self.CONFIDENCE_THRESHOLD:
                    error_msg = (
                        f"Low confidence ({confidence:.2f} < {self.CONFIDENCE_THRESHOLD}) "
                        f"— action '{raw_action}' not executed. "
                        f"Please clarify the goal or try a simpler request."
                    )
                    result.low_confidence_count += 1
                    last_failed_action = raw_action
                    result.retry_history.append({
                        'action': raw_action, 'observation': error_msg,
                        'valid': False, 'step_num': result.num_steps + 1,
                        'confidence': confidence,
                    })
                    continue
                
                try:
                    name, args = parse_action(raw_action)
                    valid, msg = validate_action(name, args)
                    if not valid:
                        error_msg = f"Validation error: {msg}"
                        last_failed_action = raw_action
                        result.retry_history.append({
                            'action': raw_action, 'observation': error_msg,
                            'valid': False, 'step_num': result.num_steps + 1,
                            'confidence': confidence,
                        })
                        continue
                    
                    action_text = raw_action
                    action_valid = True
                    action_confidence = confidence
                    break
                except ActionParseError as e:
                    error_msg = f"Parse error: {str(e)}"
                    last_failed_action = raw_action
                    result.retry_history.append({
                        'action': raw_action, 'observation': error_msg,
                        'valid': False, 'step_num': result.num_steps + 1,
                        'confidence': confidence,
                    })
                    continue
            
            if not action_valid:
                result.aborted = True
                result.abort_reason = f"Action generation failed after retries: {error_msg}"
                result.steps.append({
                    'action': last_failed_action,
                    'observation': error_msg,
                    'valid': False,
                    'step_num': result.num_steps + 1,
                    'confidence': confidence if 'confidence' in locals() else 0.0,
                })
                break
            
            # Log confidence
            result.confidence_scores.append(action_confidence)
            if action_confidence < self.CONFIDENCE_WARNING:
                # Still execute, but log a warning
                pass

            observation, is_terminal = self.env.step(action_text)
            
            step_record = {
                'action': action_text,
                'observation': observation,
                'valid': action_valid,
                'step_num': result.num_steps + 1,
                'confidence': action_confidence,
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
    
    def _generate_action(self, prompt: str, temperature: float = 0.0, top_k: int = None) -> tuple[str, float]:
        """
        Generate an action with confidence score.
        Returns (action_text, mean_confidence).
        """
        k = top_k if top_k is not None else self._gen_top_k
        raw, confidence, _ = generate_with_confidence(
            self.model, self.tokenizer, prompt,
            max_new_tokens=self._gen_max_tokens,
            temperature=temperature,
            top_k=k,
            repetition_penalty=1.0,  # CRITICAL: 1.0 allows copying exact task IDs and titles from context
            stop_sequences=["\n", "\n### Observation", "### Observation"],
        )
        return self._clean_action(raw), confidence
    
    def _detect_loop(self, history: list[dict]) -> bool:
        """
        Detects repetitive action loops:
        1. 3 identical consecutive actions (A -> A -> A)
        2. 2-step alternating cycle repeated twice (A -> B -> A -> B)
        3. 3-step cycle repeated twice (A -> B -> C -> A -> B -> C)
        """
        if not history or len(history) < 3:
            return False

        actions = [h.get('action', '').strip() for h in history if isinstance(h, dict) and 'action' in h]
        if len(actions) < 3:
            return False

        # 1-step repeat: A -> A -> A
        if len(actions) >= 3 and actions[-1] and actions[-1] == actions[-2] == actions[-3]:
            return True

        # 2-step alternating cycle: A -> B -> A -> B
        if len(actions) >= 4 and actions[-1] and actions[-2]:
            if actions[-4] == actions[-2] and actions[-3] == actions[-1]:
                return True

        # 3-step repeating cycle: A -> B -> C -> A -> B -> C
        if len(actions) >= 6 and all(actions[-i] for i in range(1, 4)):
            if actions[-6:-3] == actions[-3:]:
                return True

        return False
    
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
