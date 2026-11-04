from __future__ import annotations
try:
    import fcntl
except ImportError:
    fcntl = None
import json
import os
import tempfile
from dataclasses import dataclass, asdict
from typing import Optional
import datetime
import logging

try:
    from kanha.utils.logging import get_logger
except ImportError:
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            logger.addHandler(logging.NullHandler())
        return logger

logger = get_logger(__name__)

@dataclass
class Task:
    id: int
    title: str
    due: Optional[str] = None
    priority: str = 'med'
    tag: Optional[str] = None
    status: str = 'todo'

class TaskStore:
    # Maximum number of undo operations to keep (configurable, no hardcoded magic number)
    MAX_UNDO_DEPTH = 10

    def __init__(self, path='data/lam/tasks.json', reference_date=None, list_limit=None, auto_save: bool = True):
        self.path = path
        self.reference_date = reference_date if reference_date else datetime.date.today()
        self.auto_save = auto_save
        self._next_id = 1
        self._tasks: dict[int, Task] = {}
        self._undo_stack: list[tuple[str, dict]] = []

        # Display limit for list()/find() — read from config if available, else default 6
        if list_limit is not None:
            self._list_limit = list_limit
        else:
            try:
                from kanha.utils.config import cfg
                self._list_limit = getattr(cfg.lam, 'list_display_limit', 6)
            except Exception:
                self._list_limit = 6

        self.load()

    def add(self, title: str, due: Optional[str] = None, prio: str = 'med', tag: Optional[str] = None) -> Task:
        task = Task(id=self._next_id, title=title, due=due, priority=prio, tag=tag, status='todo')
        self._tasks[self._next_id] = task
        self._next_id += 1
        self._push_undo('add', {'id': task.id})
        self.save()
        return task

    def _matches(self, task: Task, due=None, prio=None, tag=None, status=None) -> bool:
        if status is not None:
            if task.status.lower() != status.lower():
                return False
        else:
            if task.status != 'todo':
                return False
        
        if due is not None:
            if task.due is None:
                return False
            resolved_query = self.resolve_date(due).lower()
            resolved_task = self.resolve_date(task.due).lower()
            if task.due.lower() != due.lower() and resolved_task != resolved_query:
                return False
        if prio is not None and task.priority.lower() != prio.lower():
            return False
        if tag is not None and (task.tag is None or task.tag.lower() != tag.lower()):
            return False
        return True

    def list(self, due=None, prio=None, tag=None, status=None, limit=None) -> tuple[list[Task], int]:
        if limit is None:
            limit = self._list_limit
        matches = []
        for task in self._tasks.values():
            if task.status != 'deleted' and self._matches(task, due, prio, tag, status):
                matches.append(task)
        
        return matches[:limit], max(0, len(matches) - limit)

    def find(self, query: str) -> list[Task]:
        matches = []
        query_lower = query.lower()
        for task in self._tasks.values():
            if task.status != 'deleted' and query_lower in task.title.lower():
                matches.append(task)
        return matches[:self._list_limit]

    def count(self, due=None, prio=None, tag=None, status=None) -> int:
        c = 0
        for task in self._tasks.values():
            if task.status != 'deleted' and self._matches(task, due, prio, tag, status):
                c += 1
        return c

    def edit(self, task_id: int, field: str, value: str) -> Task:
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        task = self._tasks[task_id]
        if task.status == 'deleted':
            raise ValueError(f"Task {task_id} is deleted")
        
        valid_fields = {'title', 'due', 'priority', 'tag', 'prio'}
        if field not in valid_fields:
            raise ValueError(f"Invalid field: {field}")
            
        if field == 'prio':
            field = 'priority'
            
        old_value = getattr(task, field)
        setattr(task, field, value)
        
        self._push_undo('edit', {'id': task_id, 'field': field, 'old_value': old_value})
        self.save()
        return task

    def done(self, task_id: int) -> Task:
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        task = self._tasks[task_id]
        if task.status == 'deleted':
            raise ValueError(f"Task {task_id} is deleted")
        if task.status == 'done':
            raise ValueError(f"Task {task_id} is already done")
            
        task.status = 'done'
        self._push_undo('done', {'ids': [task_id]})
        self.save()
        return task

    def done_all(self, due=None, prio=None, tag=None) -> int:
        changed_ids = []
        for task in self._tasks.values():
            if task.status == 'todo' and self._matches(task, due, prio, tag, status='todo'):
                task.status = 'done'
                changed_ids.append(task.id)
                
        if changed_ids:
            self._push_undo('done_all', {'ids': changed_ids})
            self.save()
        return len(changed_ids)

    def delete(self, task_id: int) -> Task:
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        task = self._tasks[task_id]
        if task.status == 'deleted':
            raise ValueError(f"Task {task_id} already deleted")
            
        old_status = task.status
        task.status = 'deleted'
        self._push_undo('delete', {'ids': [task_id], 'old_status': old_status})
        self.save()
        return task

    def delete_done(self) -> int:
        deleted_ids = []
        for task in self._tasks.values():
            if task.status == 'done':
                task.status = 'deleted'
                deleted_ids.append(task.id)
                
        if deleted_ids:
            self._push_undo('delete_done', {'ids': deleted_ids})
            self.save()
        return len(deleted_ids)

    def _push_undo(self, operation: str, data: dict):
        self._undo_stack.append((operation, data))
        # Cap undo history to prevent unbounded memory growth
        if len(self._undo_stack) > self.MAX_UNDO_DEPTH:
            self._undo_stack = self._undo_stack[-self.MAX_UNDO_DEPTH:]

    def undo(self) -> str:
        if not self._undo_stack:
            raise ValueError("Nothing to undo")
            
        op, data = self._undo_stack.pop()
        msg = ""
        
        if op == 'add':
            task_id = data['id']
            if task_id in self._tasks:
                del self._tasks[task_id]
                if self._next_id == task_id + 1:
                    self._next_id = task_id
            msg = f"Undid add: removed task {task_id}"
        elif op == 'edit':
            task_id = data['id']
            if task_id in self._tasks:
                setattr(self._tasks[task_id], data['field'], data['old_value'])
            msg = f"Undid edit: restored {data['field']} of task {task_id}"
        elif op in ('done', 'done_all'):
            for tid in data['ids']:
                if tid in self._tasks:
                    self._tasks[tid].status = 'todo'
            msg = f"Undid done: marked {len(data['ids'])} tasks as todo"
        elif op == 'delete':
            for tid in data['ids']:
                if tid in self._tasks:
                    self._tasks[tid].status = data.get('old_status', 'todo')
            msg = f"Undid delete: restored {len(data['ids'])} tasks"
        elif op == 'delete_done':
            for tid in data['ids']:
                if tid in self._tasks:
                    self._tasks[tid].status = 'done'
            msg = f"Undid delete: restored {len(data['ids'])} tasks"
            
        self.save()
        return msg

    def snapshot(self) -> dict:
        return {
            str(tid): asdict(task) 
            for tid, task in self._tasks.items() 
            if task.status != 'deleted'
        }

    def reset(self):
        self._tasks.clear()
        self._undo_stack.clear()
        self._next_id = 1
        self.save()

    def save(self, force: bool = False):
        if not self.auto_save and not force:
            return
        parent_dir = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent_dir, exist_ok=True)
        data = {
            'next_id': self._next_id,
            'tasks': {str(k): asdict(v) for k, v in self._tasks.items()}
        }
        # File-based locking for multi-worker safety (uvicorn --workers > 1)
        lock_path = self.path + '.lock'
        if fcntl is not None:
            with open(lock_path, 'a') as lock_f:
                fcntl.flock(lock_f, fcntl.LOCK_EX)
                fd, temp_path = tempfile.mkstemp(dir=parent_dir)
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2)
                os.replace(temp_path, self.path)
        else:
            fd, temp_path = tempfile.mkstemp(dir=parent_dir)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.path)

    def load(self):
        if os.path.exists(self.path):
            try:
                if fcntl is not None:
                    lock_path = self.path + '.lock'
                    with open(lock_path, 'a') as lock_f:
                        fcntl.flock(lock_f, fcntl.LOCK_SH)
                        with open(self.path, 'r') as f:
                            data = json.load(f)
                else:
                    with open(self.path, 'r') as f:
                        data = json.load(f)
                self._next_id = data.get('next_id', 1)
                self._tasks = {
                    int(k): Task(**v) for k, v in data.get('tasks', {}).items()
                }
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to load tasks from {self.path} (corrupted or empty): {e}. Initializing empty store.")
                self._next_id = 1
                self._tasks = {}

    @staticmethod
    def resolve_symbolic_date(symbolic: str, reference_date=None) -> str:
        """Resolves symbolic date strings (today, tomorrow, monday, etc.)
        to ISO format dates relative to reference_date."""
        if not symbolic:
            return symbolic
        symbolic_lower = str(symbolic).lower().strip()

        if reference_date is None:
            ref = datetime.date.today()
        elif isinstance(reference_date, str):
            try:
                ref = datetime.date.fromisoformat(reference_date)
            except ValueError:
                return symbolic
        else:
            ref = reference_date

        day_map = {
            'monday': 0, 'tuesday': 1, 'wednesday': 2,
            'thursday': 3, 'friday': 4, 'saturday': 5, 'sunday': 6,
        }

        if symbolic_lower == 'today':
            return ref.isoformat()
        elif symbolic_lower == 'tomorrow':
            return (ref + datetime.timedelta(days=1)).isoformat()
        elif symbolic_lower == 'next week':
            # Next Monday
            days_until_monday = (7 - ref.weekday()) % 7
            if days_until_monday == 0:
                days_until_monday = 7
            return (ref + datetime.timedelta(days=days_until_monday)).isoformat()
        elif symbolic_lower in day_map:
            target = day_map[symbolic_lower]
            current = ref.weekday()
            days_ahead = (target - current) % 7
            if days_ahead == 0:
                days_ahead = 7  # next occurrence, not today
            return (ref + datetime.timedelta(days=days_ahead)).isoformat()
        else:
            # Return as-is if not a recognized symbolic date
            return symbolic

    def resolve_date(self, symbolic: str) -> str:
        """Resolves symbolic date strings (today, tomorrow, monday, etc.)
        to ISO format dates relative to reference_date."""
        return self.resolve_symbolic_date(symbolic, self.reference_date)
