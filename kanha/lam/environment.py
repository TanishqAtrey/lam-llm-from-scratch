from __future__ import annotations
from typing import Optional
from kanha.lam.task_store import TaskStore, Task
from kanha.lam.actions import parse_action, validate_action, execute_action, ActionParseError

class TaskEnvironment:
    def __init__(self, reference_date=None, task_store_path=None, auto_save: bool = True):
        # Read defaults from config if not provided
        try:
            from kanha.utils.config import cfg
            lam_cfg = cfg.lam
            if task_store_path is None:
                task_store_path = getattr(lam_cfg, 'task_store_path', 'data/lam/tasks.json')
            if reference_date is None:
                reference_date = getattr(lam_cfg, 'reference_date', None)
        except Exception:
            if task_store_path is None:
                task_store_path = 'data/lam/tasks.json'

        self.store = TaskStore(path=task_store_path, reference_date=reference_date, auto_save=auto_save)
        self._history: list[dict] = []

    def reset(self, initial_tasks: Optional[list[dict]] = None):
        self.store.reset()
        self._history = []
        if initial_tasks:
            for i, task_data in enumerate(initial_tasks, start=1):
                due = task_data.get('due')
                task = Task(
                    id=i,
                    title=task_data['title'],
                    due=due,
                    priority=task_data.get('prio', task_data.get('priority', 'med')),
                    tag=task_data.get('tag'),
                    status=task_data.get('status', 'todo'),
                )
                self.store._tasks[i] = task
            self.store._next_id = len(initial_tasks) + 1
            self.store._undo_stack.clear()
            self.store.save()

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
