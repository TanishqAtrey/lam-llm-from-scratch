from __future__ import annotations
import re
from typing import Any
from kanha.lam.task_store import TaskStore, Task

class ActionParseError(Exception):
    pass

ACTION_SCHEMA = {
    'add': {'req': ['title'], 'opt': ['due', 'prio', 'tag']},
    'list': {'req': [], 'opt': ['due', 'prio', 'tag', 'status']},
    'find': {'req': ['q'], 'opt': []},
    'count': {'req': [], 'opt': ['due', 'prio', 'tag', 'status']},
    'edit': {'req': ['id', 'field', 'value'], 'opt': []},
    'done': {'req': ['id'], 'opt': []},
    'done_all': {'req': [], 'opt': ['due', 'prio', 'tag']},
    'delete': {'req': ['id'], 'opt': []},
    'delete_done': {'req': [], 'opt': []},
    'undo': {'req': [], 'opt': []},
    'ask': {'req': ['msg'], 'opt': []},
    'finish': {'req': ['msg'], 'opt': []}
}

def parse_action(text: str) -> tuple[str, dict]:
    text = text.strip()
    match = re.match(r'^([a-zA-Z_]+)\((.*)\)$', text, re.DOTALL)
    if not match:
        raise ActionParseError("Malformed action format")
    
    action_name = match.group(1)
    args_str = match.group(2).strip()
    args_dict = {}
    
    if not args_str:
        return action_name, args_dict
        
    state = 'key'
    current_key = ''
    current_val = ''
    in_quotes = False
    quote_char = ''
    
    for i, char in enumerate(args_str):
        if state == 'key':
            if char == '=':
                state = 'value'
                current_key = current_key.strip()
            else:
                current_key += char
        elif state == 'value':
            if not in_quotes:
                if char in ('"', "'"):
                    if not current_val.strip():
                        in_quotes = True
                        quote_char = char
                    else:
                        current_val += char
                elif char == ',':
                    args_dict[current_key] = current_val.strip()
                    current_key = ''
                    current_val = ''
                    state = 'key'
                else:
                    current_val += char
            else:
                if char == quote_char:
                    in_quotes = False
                else:
                    current_val += char
                    
    if state == 'key' and current_key.strip():
        raise ActionParseError("Malformed arguments: missing '='")
    if in_quotes:
        raise ActionParseError("Malformed arguments: unclosed quote")
    current_key = current_key.strip()
    if current_key:
        args_dict[current_key] = current_val.strip()
        
    return action_name, args_dict

def validate_action(name: str, args: dict) -> tuple[bool, str]:
    if name not in ACTION_SCHEMA:
        return False, f"Unknown action: {name}"
        
    schema = ACTION_SCHEMA[name]
    for req_arg in schema['req']:
        if req_arg not in args:
            return False, f"Missing required argument: {req_arg}"
            
    for arg in args:
        if arg not in schema['req'] and arg not in schema['opt']:
            return False, f"Unknown argument: {arg}"
            
    if 'id' in args:
        try:
            int(args['id'])
        except ValueError:
            return False, "Argument 'id' must be an integer"
            
    if 'prio' in args and args['prio'] not in {'low', 'med', 'high'}:
        return False, "Argument 'prio' must be one of: low, med, high"
        
    if 'field' in args and args['field'] not in {'title', 'due', 'priority', 'tag'}:
        return False, "Argument 'field' must be one of: title, due, priority, tag"
        
    if 'status' in args and args['status'] not in {'todo', 'done'}:
        return False, "Argument 'status' must be one of: todo, done"
        
    return True, ""

def format_task(task: Task) -> str:
    parts = [f"#{task.id} {task.title}"]
    if task.due:
        parts.append(task.due)
    if task.priority:
        parts.append(task.priority)
    if task.tag:
        parts.append(task.tag)
    return " | ".join(parts)

def format_task_list(tasks: list[Task], remaining: int) -> str:
    if not tasks:
        return "No tasks found."
    lines = [format_task(t) for t in tasks]
    if remaining > 0:
        lines.append(f"(+{remaining} more)")
    return "\n".join(lines)

def execute_action(name: str, args: dict, store: TaskStore) -> str:
    try:
        if name == 'add':
            t = store.add(title=args['title'], due=args.get('due'), prio=args.get('prio', 'med'), tag=args.get('tag'))
            return f"Added: {format_task(t)}"
        elif name == 'list':
            tasks, rem = store.list(due=args.get('due'), prio=args.get('prio'), tag=args.get('tag'), status=args.get('status'))
            return format_task_list(tasks, rem)
        elif name == 'find':
            tasks = store.find(args['q'])
            return format_task_list(tasks, 0)
        elif name == 'count':
            c = store.count(due=args.get('due'), prio=args.get('prio'), tag=args.get('tag'), status=args.get('status'))
            return f"Count: {c}"
        elif name == 'edit':
            t = store.edit(int(args['id']), args['field'], args['value'])
            return f"Edited: {format_task(t)}"
        elif name == 'done':
            t = store.done(int(args['id']))
            return f"Done: {format_task(t)}"
        elif name == 'done_all':
            c = store.done_all(due=args.get('due'), prio=args.get('prio'), tag=args.get('tag'))
            return f"Marked {c} tasks as done."
        elif name == 'delete':
            t = store.delete(int(args['id']))
            return f"Deleted: #{t.id} {t.title}"
        elif name == 'delete_done':
            c = store.delete_done()
            return f"Deleted {c} done tasks."
        elif name == 'undo':
            return store.undo()
        elif name in ('ask', 'finish'):
            return args['msg']
    except ValueError as e:
        return f"error: {str(e)}"
    
    return "error: unknown action execution"
