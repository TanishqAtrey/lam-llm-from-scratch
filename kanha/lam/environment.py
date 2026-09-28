from __future__ import annotations
from typing import Optional
from kanha.lam.task_store import TaskStore
from kanha.lam.actions import parse_action, validate_action, execute_action, ActionParseError

class TaskEnvironment:
    def __init__(self, reference_date=None, task_store_path='data/lam/tasks.json'):
        self.store = TaskStore(path=task_store_path, reference_date=reference_date)
        self._history: list[dict] = []

    def reset(self, initial_tasks: Optional[list[dict]] = None):
        self.store.reset()
        self._history = []
        if initial_tasks:
            for task_data in initial_tasks:
                self.store.add(
                    title=task_data['title'],
                    due=task_data.get('due'),
                    prio=task_data.get('prio', 'med'),
                    tag=task_data.get('tag')
                )

    def step(self, action_text: str) -> tuple[str, bool]:
        try:
            name, args = parse_action(action_text)
        except ActionParseError:
            self._history.append({'action': action_text, 'observation': 'error: could not parse action', 'valid': False, 'terminal': False})
            return 'error: could not parse action', False

        valid, error_msg = validate_action(name, args)
        if not valid:
            obs = f"error: {error_msg}"
            self._history.append({'action': action_text, 'observation': obs, 'valid': False, 'terminal': False})
            return obs, False

        observation = execute_action(name, args, self.store)
        is_terminal = name in ('finish', 'ask')
        
        self._history.append({'action': action_text, 'observation': observation, 'valid': True, 'terminal': is_terminal})
        return observation, is_terminal

    def get_state(self) -> dict:
        return self.store.snapshot()

    def get_trajectory(self) -> list[dict]:
        return self._history
