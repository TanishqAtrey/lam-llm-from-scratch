"""
kanha/lam/langchain/tools.py
LangChain StructuredTool wrappers for Kanha Tasks actions.

Each of the 12 LAM actions is wrapped as a LangChain tool so that:
 1. They are inspectable via LangChain's tool-listing API.
 2. LangGraph nodes can call them through a unified interface.
 3. Future upgrades to proper tool-calling LLMs slot in seamlessly.

The small Kanha model does NOT use LangChain's tool-calling protocol;
it emits raw action strings like ``add(title="buy milk")``.  The graph
parses those and dispatches to these tools via ``execute_tool_by_name()``.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool

from kanha.lam.task_store import TaskStore
from kanha.lam.actions import format_task, format_task_list


# ── Tool factories ────────────────────────────────────────────────────────────
# Each factory takes a TaskStore and returns a StructuredTool bound to it.

def _make_add_tool(store: TaskStore) -> StructuredTool:
    def add_task(title: str, due: Optional[str] = None,
                 prio: str = "med", tag: Optional[str] = None) -> str:
        """Add a new task to the task list."""
        t = store.add(title=title, due=due, prio=prio, tag=tag)
        return f"Added: {format_task(t)}"
    return StructuredTool.from_function(add_task, name="add",
                                        description="Add a new task")


def _make_list_tool(store: TaskStore) -> StructuredTool:
    def list_tasks(due: Optional[str] = None, prio: Optional[str] = None,
                   tag: Optional[str] = None, status: Optional[str] = None) -> str:
        """List tasks, optionally filtered by due date, priority, tag, or status."""
        tasks, rem = store.list(due=due, prio=prio, tag=tag, status=status)
        return format_task_list(tasks, rem)
    return StructuredTool.from_function(list_tasks, name="list",
                                        description="List tasks with optional filters")


def _make_find_tool(store: TaskStore) -> StructuredTool:
    def find_tasks(q: str) -> str:
        """Search tasks by title substring."""
        tasks = store.find(q)
        return format_task_list(tasks, 0)
    return StructuredTool.from_function(find_tasks, name="find",
                                        description="Search tasks by title")


def _make_count_tool(store: TaskStore) -> StructuredTool:
    def count_tasks(due: Optional[str] = None, prio: Optional[str] = None,
                    tag: Optional[str] = None, status: Optional[str] = None) -> str:
        """Count matching tasks."""
        c = store.count(due=due, prio=prio, tag=tag, status=status)
        return f"Count: {c}"
    return StructuredTool.from_function(count_tasks, name="count",
                                        description="Count matching tasks")


def _make_edit_tool(store: TaskStore) -> StructuredTool:
    def edit_task(id: int, field: str, value: str) -> str:
        """Edit a field of an existing task."""
        try:
            t = store.edit(id, field, value)
            return f"Edited: {format_task(t)}"
        except ValueError as e:
            return f"error: {e}"
    return StructuredTool.from_function(edit_task, name="edit",
                                        description="Edit a task field")


def _make_done_tool(store: TaskStore) -> StructuredTool:
    def done_task(id: int) -> str:
        """Mark a task as done."""
        try:
            t = store.done(id)
            return f"Done: {format_task(t)}"
        except ValueError as e:
            return f"error: {e}"
    return StructuredTool.from_function(done_task, name="done",
                                        description="Mark a task as done")


def _make_done_all_tool(store: TaskStore) -> StructuredTool:
    def done_all_tasks(due: Optional[str] = None, prio: Optional[str] = None,
                       tag: Optional[str] = None) -> str:
        """Mark all matching tasks as done."""
        c = store.done_all(due=due, prio=prio, tag=tag)
        return f"Marked {c} tasks as done."
    return StructuredTool.from_function(done_all_tasks, name="done_all",
                                        description="Mark all matching tasks as done")


def _make_delete_tool(store: TaskStore) -> StructuredTool:
    def delete_task(id: int) -> str:
        """Soft-delete a task."""
        try:
            t = store.delete(id)
            return f"Deleted: #{t.id} {t.title}"
        except ValueError as e:
            return f"error: {e}"
    return StructuredTool.from_function(delete_task, name="delete",
                                        description="Delete a task")


def _make_delete_done_tool(store: TaskStore) -> StructuredTool:
    def delete_done_tasks() -> str:
        """Delete all completed tasks."""
        c = store.delete_done()
        return f"Deleted {c} done tasks."
    return StructuredTool.from_function(delete_done_tasks, name="delete_done",
                                        description="Delete all completed tasks")


def _make_undo_tool(store: TaskStore) -> StructuredTool:
    def undo_action() -> str:
        """Undo the last action."""
        try:
            return store.undo()
        except ValueError as e:
            return f"error: {e}"
    return StructuredTool.from_function(undo_action, name="undo",
                                        description="Undo the last action")


def _make_ask_tool() -> StructuredTool:
    def ask_user(msg: str) -> str:
        """Ask the user a clarifying question (terminal action)."""
        return msg
    return StructuredTool.from_function(ask_user, name="ask",
                                        description="Ask a clarifying question")


def _make_finish_tool() -> StructuredTool:
    def finish_task(msg: str) -> str:
        """Complete the goal with a final message (terminal action)."""
        return msg
    return StructuredTool.from_function(finish_task, name="finish",
                                        description="Complete the goal with a message")


# ── Public API ────────────────────────────────────────────────────────────────

def build_task_tools(store: TaskStore) -> dict[str, StructuredTool]:
    """Build all 12 LAM tools bound to the given TaskStore.

    Returns:
        A dict mapping action name → StructuredTool, for easy
        lookup during action dispatch.
    """
    tools = {
        "add":         _make_add_tool(store),
        "list":        _make_list_tool(store),
        "find":        _make_find_tool(store),
        "count":       _make_count_tool(store),
        "edit":        _make_edit_tool(store),
        "done":        _make_done_tool(store),
        "done_all":    _make_done_all_tool(store),
        "delete":      _make_delete_tool(store),
        "delete_done": _make_delete_done_tool(store),
        "undo":        _make_undo_tool(store),
        "ask":         _make_ask_tool(),
        "finish":      _make_finish_tool(),
    }
    return tools


def execute_tool_by_name(
    tools: dict[str, StructuredTool],
    action_name: str,
    action_args: dict,
) -> str:
    """Execute a LangChain tool by action name and parsed args.

    This bridges the gap between the Kanha model's raw action format
    (``add(title="buy milk")``) and LangChain's StructuredTool API.

    Args:
        tools: dict from ``build_task_tools()``
        action_name: parsed action name (e.g. "add")
        action_args: parsed keyword args dict

    Returns:
        The tool's string observation.
    """
    if action_name not in tools:
        return f"error: unknown action '{action_name}'"

    tool = tools[action_name]

    # Convert string 'id' to int if present
    if "id" in action_args:
        try:
            action_args["id"] = int(action_args["id"])
        except (ValueError, TypeError):
            return f"error: 'id' must be an integer, got '{action_args['id']}'"

    try:
        return tool.invoke(action_args)
    except Exception as e:
        return f"error: {e}"
