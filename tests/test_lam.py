"""
tests/test_lam.py
Comprehensive tests for the Kanha Tasks LAM system.

Tests cover:
- TaskStore: CRUD, undo, persistence, edge cases
- Action parser: parsing, validation, execution
- Environment: step, reset, trajectory tracking
- Trajectory: formatting, step expansion, dataset masking
- Data generation: templates, oracle, generator, split
- LangChain/LangGraph: tools, model wrapper, graph routing
"""

import dataclasses
import json
import os

import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kanha.lam.task_store import TaskStore, Task
from kanha.lam.actions import (
    parse_action,
    validate_action,
    execute_action,
    format_task,
    format_task_list,
    ActionParseError,
    ACTION_SCHEMA,
)
from kanha.lam.environment import TaskEnvironment
from kanha.lam.trajectory import (
    format_trajectory,
    format_step_prompt,
    expand_to_steps,
    GOAL_MARKER,
    ACTION_MARKER,
    OBSERVATION_MARKER,
)
from kanha.prompting.builder import (
    LAM_GOAL_MARKER,
    LAM_ACTION_MARKER,
    LAM_OBSERVATION_MARKER,
    build_lam_step_prompt,
    build_lam_training_pair,
)


# ═══════════════════════════════════════════════════════════════════════════════
# TaskStore Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTaskStore:
    """Tests for kanha.lam.task_store.TaskStore."""

    def _make_store(self, tmpdir):
        path = os.path.join(tmpdir, "tasks.json")
        return TaskStore(path=path)

    def test_add_and_list(self, tmp_path):
        store = self._make_store(str(tmp_path))
        t = store.add("buy milk", due="today", prio="high", tag="errands")
        assert t.id == 1
        assert t.title == "buy milk"
        assert t.due == "today"
        assert t.priority == "high"
        assert t.tag == "errands"
        assert t.status == "todo"

        tasks, remaining = store.list()
        assert len(tasks) == 1
        assert remaining == 0
        assert tasks[0].title == "buy milk"

    def test_add_defaults(self, tmp_path):
        store = self._make_store(str(tmp_path))
        t = store.add("task1")
        assert t.priority == "med"
        assert t.due is None
        assert t.tag is None

    def test_list_filters(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk", due="today", prio="high", tag="errands")
        store.add("call dentist", due="tomorrow", prio="med", tag="health")
        store.add("pay bills", due="today", prio="low", tag="finance")

        # Filter by due
        tasks, _ = store.list(due="today")
        assert len(tasks) == 2
        titles = {t.title for t in tasks}
        assert "buy milk" in titles
        assert "pay bills" in titles

        # Filter by prio
        tasks, _ = store.list(prio="high")
        assert len(tasks) == 1
        assert tasks[0].title == "buy milk"

        # Filter by tag
        tasks, _ = store.list(tag="health")
        assert len(tasks) == 1
        assert tasks[0].title == "call dentist"

    def test_list_limit(self, tmp_path):
        store = self._make_store(str(tmp_path))
        for i in range(10):
            store.add(f"task {i}")

        tasks, remaining = store.list(limit=6)
        assert len(tasks) == 6
        assert remaining == 4

    def test_find(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        store.add("buy bread")
        store.add("call mom")

        results = store.find("buy")
        assert len(results) == 2
        titles = {t.title for t in results}
        assert "buy milk" in titles
        assert "buy bread" in titles

    def test_find_case_insensitive(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("Buy Milk")
        results = store.find("buy milk")
        assert len(results) == 1

    def test_count(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("task1", due="today")
        store.add("task2", due="today")
        store.add("task3", due="tomorrow")

        assert store.count() == 3
        assert store.count(due="today") == 2
        assert store.count(due="tomorrow") == 1

    def test_edit(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk", due="today")

        t = store.edit(1, "due", "tomorrow")
        assert t.due == "tomorrow"

        t = store.edit(1, "title", "buy almond milk")
        assert t.title == "buy almond milk"

    def test_edit_invalid_field(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")

        with pytest.raises(ValueError, match="Invalid field"):
            store.edit(1, "invalid_field", "value")

    def test_edit_missing_task(self, tmp_path):
        store = self._make_store(str(tmp_path))

        with pytest.raises(ValueError, match="not found"):
            store.edit(99, "title", "new title")

    def test_done(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")

        t = store.done(1)
        assert t.status == "done"

    def test_done_already_done(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        store.done(1)

        with pytest.raises(ValueError, match="already done"):
            store.done(1)

    def test_done_all(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("task1", due="today")
        store.add("task2", due="today")
        store.add("task3", due="tomorrow")

        count = store.done_all(due="today")
        assert count == 2
        assert store.count(status="done") == 2
        assert store.count(due="tomorrow") == 1

    def test_delete(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        store.delete(1)

        tasks, _ = store.list()
        assert len(tasks) == 0

    def test_delete_done(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("task1")
        store.add("task2")
        store.done(1)

        count = store.delete_done()
        assert count == 1
        tasks, _ = store.list()
        assert len(tasks) == 1
        assert tasks[0].title == "task2"

    def test_undo_add(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        assert store.count() == 1

        msg = store.undo()
        assert "Undid" in msg
        assert store.count() == 0

    def test_undo_edit(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk", due="today")
        store.edit(1, "due", "tomorrow")

        store.undo()
        tasks, _ = store.list()
        assert tasks[0].due == "today"

    def test_undo_done(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        store.done(1)
        assert store.count(status="done") == 1

        store.undo()
        assert store.count() == 1
        tasks, _ = store.list()
        assert tasks[0].status == "todo"

    def test_undo_delete(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("buy milk")
        store.delete(1)
        assert store.count() == 0

        store.undo()
        assert store.count() == 1

    def test_undo_empty_stack(self, tmp_path):
        store = self._make_store(str(tmp_path))
        with pytest.raises(ValueError, match="Nothing to undo"):
            store.undo()

    def test_snapshot(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("task1")
        store.add("task2")
        store.delete(2)

        snap = store.snapshot()
        assert len(snap) == 1
        assert "1" in snap

    def test_reset(self, tmp_path):
        store = self._make_store(str(tmp_path))
        store.add("task1")
        store.add("task2")
        store.reset()

        assert store.count() == 0
        assert store._next_id == 1

    def test_persistence(self, tmp_path):
        path = os.path.join(str(tmp_path), "tasks.json")
        store1 = TaskStore(path=path)
        store1.add("persisted task", due="tomorrow")

        store2 = TaskStore(path=path)
        tasks, _ = store2.list()
        assert len(tasks) == 1
        assert tasks[0].title == "persisted task"

    def test_atomic_save(self, tmp_path):
        path = os.path.join(str(tmp_path), "tasks.json")
        store = TaskStore(path=path)
        store.add("test")
        assert os.path.exists(path)

        with open(path) as f:
            data = json.load(f)
        assert "tasks" in data
        assert "next_id" in data


# ═══════════════════════════════════════════════════════════════════════════════
# Action Parser Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestActionParser:
    """Tests for kanha.lam.actions parsing and validation."""

    def test_parse_simple_add(self):
        name, args = parse_action('add(title="buy milk")')
        assert name == "add"
        assert args["title"] == "buy milk"

    def test_parse_add_with_all_args(self):
        name, args = parse_action('add(title="buy milk", due=tomorrow, prio=high, tag=errands)')
        assert name == "add"
        assert args["title"] == "buy milk"
        assert args["due"] == "tomorrow"
        assert args["prio"] == "high"
        assert args["tag"] == "errands"

    def test_parse_empty_args(self):
        name, args = parse_action("undo()")
        assert name == "undo"
        assert args == {}

    def test_parse_delete_done(self):
        name, args = parse_action("delete_done()")
        assert name == "delete_done"
        assert args == {}

    def test_parse_done_with_id(self):
        name, args = parse_action("done(id=3)")
        assert name == "done"
        assert args["id"] == "3"

    def test_parse_find(self):
        name, args = parse_action('find(q="buy milk")')
        assert name == "find"
        assert args["q"] == "buy milk"

    def test_parse_finish(self):
        name, args = parse_action('finish(msg="Task added.")')
        assert name == "finish"
        assert args["msg"] == "Task added."

    def test_parse_edit(self):
        name, args = parse_action('edit(id=1, field=due, value=friday)')
        assert name == "edit"
        assert args["id"] == "1"
        assert args["field"] == "due"
        assert args["value"] == "friday"

    def test_parse_quoted_value_with_comma(self):
        name, args = parse_action('add(title="buy eggs, bread")')
        assert args["title"] == "buy eggs, bread"

    def test_parse_malformed(self):
        with pytest.raises(ActionParseError):
            parse_action("not an action")

    def test_parse_unclosed_paren(self):
        with pytest.raises(ActionParseError):
            parse_action("add(title=hello")

    def test_validate_valid_add(self):
        ok, msg = validate_action("add", {"title": "test"})
        assert ok is True
        assert msg == ""

    def test_validate_missing_required(self):
        ok, msg = validate_action("add", {})
        assert ok is False
        assert "title" in msg.lower()

    def test_validate_unknown_action(self):
        ok, msg = validate_action("fly", {"destination": "moon"})
        assert ok is False
        assert "Unknown action" in msg

    def test_validate_unknown_param(self):
        ok, msg = validate_action("add", {"title": "test", "color": "blue"})
        assert ok is False
        assert "Unknown argument" in msg

    def test_validate_bad_id(self):
        ok, msg = validate_action("done", {"id": "abc"})
        assert ok is False
        assert "integer" in msg.lower()

    def test_validate_bad_prio(self):
        ok, msg = validate_action("add", {"title": "test", "prio": "ultra"})
        assert ok is False
        assert "prio" in msg.lower()

    def test_validate_bad_field(self):
        ok, msg = validate_action("edit", {"id": "1", "field": "color", "value": "x"})
        assert ok is False

    def test_all_12_actions_in_schema(self):
        expected = {"add", "list", "find", "count", "edit", "done", "done_all",
                    "delete", "delete_done", "undo", "ask", "finish"}
        assert set(ACTION_SCHEMA.keys()) == expected


# ═══════════════════════════════════════════════════════════════════════════════
# Action Execution Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestActionExecution:
    """Tests for kanha.lam.actions.execute_action."""

    def _make_store(self, tmp_path):
        path = os.path.join(str(tmp_path), "tasks.json")
        return TaskStore(path=path)

    def test_execute_add(self, tmp_path):
        store = self._make_store(tmp_path)
        result = execute_action("add", {"title": "buy milk", "due": "today"}, store)
        assert "buy milk" in result
        assert store.count() == 1

    def test_execute_list(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("task1")
        store.add("task2")
        result = execute_action("list", {}, store)
        assert "task1" in result
        assert "task2" in result

    def test_execute_find(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("buy milk")
        store.add("buy bread")
        result = execute_action("find", {"q": "milk"}, store)
        assert "milk" in result
        assert "bread" not in result

    def test_execute_count(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("task1")
        store.add("task2")
        result = execute_action("count", {}, store)
        assert "2" in result

    def test_execute_done(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("buy milk")
        result = execute_action("done", {"id": "1"}, store)
        assert "Done" in result or "done" in result.lower()

    def test_execute_delete(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("buy milk")
        result = execute_action("delete", {"id": "1"}, store)
        assert "Deleted" in result or "deleted" in result.lower()

    def test_execute_undo(self, tmp_path):
        store = self._make_store(tmp_path)
        store.add("buy milk")
        result = execute_action("undo", {}, store)
        assert "Undid" in result

    def test_execute_finish(self, tmp_path):
        store = self._make_store(tmp_path)
        result = execute_action("finish", {"msg": "All done!"}, store)
        assert result == "All done!"

    def test_execute_ask(self, tmp_path):
        store = self._make_store(tmp_path)
        result = execute_action("ask", {"msg": "Which one?"}, store)
        assert result == "Which one?"

    def test_execute_error_bad_id(self, tmp_path):
        store = self._make_store(tmp_path)
        result = execute_action("done", {"id": "99"}, store)
        assert "error" in result.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Format Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFormatting:
    """Tests for format_task and format_task_list."""

    def test_format_task_full(self):
        t = Task(id=3, title="buy milk", due="tomorrow", priority="high", tag="errands")
        result = format_task(t)
        assert "#3 buy milk" in result
        assert "tomorrow" in result
        assert "high" in result
        assert "errands" in result

    def test_format_task_minimal(self):
        t = Task(id=1, title="task")
        result = format_task(t)
        assert "#1 task" in result

    def test_format_task_list_empty(self):
        result = format_task_list([], 0)
        assert "No tasks" in result

    def test_format_task_list_with_remaining(self):
        tasks = [Task(id=i, title=f"task{i}") for i in range(1, 4)]
        result = format_task_list(tasks, 5)
        assert "(+5 more)" in result
        assert "task1" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Environment Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEnvironment:
    """Tests for kanha.lam.environment.TaskEnvironment."""

    def test_step_valid_action(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        obs, terminal = env.step('add(title="buy milk")')
        assert "buy milk" in obs
        assert terminal is False

    def test_step_finish_is_terminal(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        obs, terminal = env.step('finish(msg="Done!")')
        assert terminal is True
        assert obs == "Done!"

    def test_step_ask_is_terminal(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        obs, terminal = env.step('ask(msg="Which one?")')
        assert terminal is True

    def test_step_invalid_parse(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        obs, terminal = env.step("not a valid action")
        assert "error" in obs.lower()
        assert terminal is False

    def test_step_invalid_validation(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        obs, terminal = env.step("add()")  # missing required 'title'
        assert "error" in obs.lower()
        assert terminal is False

    def test_reset_with_initial_tasks(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        initial = [
            {"title": "task1", "due": "today", "prio": "high", "tag": "work"},
            {"title": "task2", "due": "tomorrow"},
        ]
        env.reset(initial_tasks=initial)
        assert env.store.count() == 2

    def test_get_trajectory(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()
        env.step('add(title="milk")')
        env.step('finish(msg="ok")')

        traj = env.get_trajectory()
        assert len(traj) == 2
        assert traj[0]["action"] == 'add(title="milk")'
        assert traj[1]["terminal"] is True

    def test_get_state_snapshot(self, tmp_path):
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset([{"title": "task1"}])
        state = env.get_state()
        assert isinstance(state, dict)
        assert len(state) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory Format Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrajectory:
    """Tests for kanha.lam.trajectory formatting."""

    def test_format_trajectory(self):
        steps = [
            {"action": 'add(title="milk")', "observation": "Added: #1 milk"},
            {"action": 'finish(msg="ok")', "observation": "ok"},
        ]
        result = format_trajectory("add milk", steps)
        assert "### Goal\n" in result
        assert "add milk" in result
        assert "### Action\n" in result
        assert 'add(title="milk")' in result
        assert "### Observation\n" in result

    def test_format_step_prompt(self):
        history = [{"action": 'find(q="milk")', "observation": "#1 buy milk"}]
        result = format_step_prompt("find milk and delete it", history)
        assert result.startswith("### Goal\n")
        assert result.endswith("### Action\n")
        assert 'find(q="milk")' in result
        assert "#1 buy milk" in result

    def test_format_step_prompt_empty_history(self):
        result = format_step_prompt("add a task", [])
        assert result == "### Goal\nadd a task\n### Action\n"

    def test_expand_to_steps(self):
        traj = {
            "goal": "add milk and bread",
            "steps": [
                {"action": 'add(title="milk")', "observation": "Added: #1 milk"},
                {"action": 'add(title="bread")', "observation": "Added: #2 bread"},
                {"action": 'finish(msg="ok")', "observation": "ok"},
            ],
        }
        expanded = expand_to_steps(traj)
        assert len(expanded) == 3

        # First example has no history
        assert expanded[0]["prompt"].endswith("### Action\n")
        assert expanded[0]["response"] == 'add(title="milk")\n'

        # Second example has one history step
        assert 'add(title="milk")' in expanded[1]["prompt"]
        assert expanded[1]["response"] == 'add(title="bread")\n'


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Builder LAM Extension Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestPromptBuilderLAM:
    """Tests for LAM extensions in kanha.prompting.builder."""

    def test_markers_match(self):
        """Trajectory markers and builder markers must be identical."""
        assert LAM_GOAL_MARKER == GOAL_MARKER
        assert LAM_ACTION_MARKER == ACTION_MARKER
        assert LAM_OBSERVATION_MARKER == OBSERVATION_MARKER

    def test_build_lam_step_prompt(self):
        result = build_lam_step_prompt("add milk", [])
        assert result == "### Goal\nadd milk\n### Action\n"

    def test_build_lam_step_prompt_with_history(self):
        history = [{"action": 'find(q="x")', "observation": "no results"}]
        result = build_lam_step_prompt("find x", history)
        assert "### Goal\n" in result
        assert "### Action\n" in result
        assert "### Observation\n" in result
        assert result.endswith("### Action\n")

    def test_build_lam_training_pair(self):
        result = build_lam_training_pair("add milk", [], 'add(title="milk")')
        assert result.startswith("### Goal\nadd milk\n### Action\n")
        assert result.endswith('add(title="milk")')


# ═══════════════════════════════════════════════════════════════════════════════
# Data Generation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDataGeneration:
    """Tests for the data generation pipeline."""

    def test_templates_have_all_tiers(self):
        from kanha.lam.data.templates import TEMPLATES
        for i in range(1, 9):
            assert f"tier{i}" in TEMPLATES, f"tier{i} missing from TEMPLATES"
            assert len(TEMPLATES[f"tier{i}"]) >= 3, f"tier{i} has too few templates"

    def test_sample_goal_tier1(self):
        import random
        from kanha.lam.data.templates import sample_goal
        rng = random.Random(42)
        goal, info = sample_goal("tier1", None, rng)
        assert isinstance(goal, str)
        assert len(goal) > 0
        assert info["tier"] == 1

    def test_sample_goal_tier8(self):
        import random
        from kanha.lam.data.templates import sample_goal
        rng = random.Random(42)
        goal, info = sample_goal("tier8", None, rng)
        assert isinstance(goal, str)
        assert info["tier"] == 8

    def test_oracle_tier1(self, tmp_path):
        from kanha.lam.data.oracle import Oracle
        oracle = Oracle()
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()

        info = {
            "solver": "_solve_direct_add",
            "slot_values": {"title": "buy milk"},
            "goal_text": "add buy milk for today",
            "due_phrase_val": "today",
        }
        steps = oracle.solve(info, env)
        assert len(steps) >= 1
        assert "add(" in steps[0][0]
        assert "buy milk" in steps[0][1]

    def test_oracle_tier8_out_of_scope(self, tmp_path):
        from kanha.lam.data.oracle import Oracle
        oracle = Oracle()
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()

        info = {
            "solver": "_solve_out_of_scope",
            "goal_text": "book me a flight",
        }
        steps = oracle.solve(info, env)
        assert len(steps) == 1
        assert "finish(" in steps[0][0]
        assert "Sorry" in steps[0][1]

    def test_generator_small_batch(self, tmp_path):
        from kanha.lam.data.generator import TrajectoryGenerator
        output = os.path.join(str(tmp_path), "data", "test.jsonl")
        gen = TrajectoryGenerator(num_trajectories=20, seed=42)
        stats = gen.generate(output)

        assert stats["total"] > 0
        assert os.path.exists(output)

        with open(output) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == stats["total"]

        # Verify structure of each trajectory
        for traj in lines:
            assert "goal" in traj
            assert "steps" in traj
            assert isinstance(traj["steps"], list)
            assert len(traj["steps"]) >= 1
            for step in traj["steps"]:
                assert "action" in step
                assert "observation" in step

    def test_split_data(self, tmp_path):
        from kanha.lam.data.generator import TrajectoryGenerator, split_data

        output = os.path.join(str(tmp_path), "data", "all.jsonl")
        gen = TrajectoryGenerator(num_trajectories=50, seed=42)
        gen.generate(output)

        train = os.path.join(str(tmp_path), "data", "train.jsonl")
        val = os.path.join(str(tmp_path), "data", "val.jsonl")
        test = os.path.join(str(tmp_path), "data", "test.jsonl")
        split_data(output, train, val, test, val_ratio=0.2, test_ratio=0.2, seed=42)

        for path in [train, val, test]:
            assert os.path.exists(path)

        with open(train) as f:
            train_lines = [l for l in f if l.strip()]
        with open(val) as f:
            val_lines = [l for l in f if l.strip()]
        with open(test) as f:
            test_lines = [l for l in f if l.strip()]

        total = len(train_lines) + len(val_lines) + len(test_lines)
        assert total > 0


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-End Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """End-to-end integration tests."""

    def test_full_trajectory_roundtrip(self, tmp_path):
        """Generate a trajectory, format it, expand it, verify masking."""
        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset()

        # Execute a sequence manually
        env.step('add(title="buy milk", due=today, prio=high)')
        env.step('finish(msg="Added milk.")')

        traj = env.get_trajectory()
        assert len(traj) == 2

        # Format it
        full = format_trajectory("add milk for today", traj)
        assert "### Goal\nadd milk for today\n" in full
        assert "### Action\n" in full

        # Expand to steps
        traj_dict = {
            "goal": "add milk for today",
            "steps": traj,
        }
        expanded = expand_to_steps(traj_dict)
        assert len(expanded) == 2

        # First step has no history
        prompt0 = expanded[0]["prompt"]
        assert prompt0.count("### Action\n") == 1  # just the trailing marker

        # Second step has one history item
        prompt1 = expanded[1]["prompt"]
        assert prompt1.count("### Action\n") == 2  # history + trailing

    def test_environment_with_oracle(self, tmp_path):
        """Verify that the oracle produces valid trajectories end-to-end."""
        from kanha.lam.data.oracle import Oracle

        env = TaskEnvironment(task_store_path=os.path.join(str(tmp_path), "t.json"))
        env.reset([{"title": "buy milk", "due": "today", "prio": "high"}])

        oracle = Oracle()

        # Test a lookup-then-act trajectory
        info = {
            "solver": "_solve_lookup_then_act",
            "slot_values": {"query": "milk"},
            "goal_text": "mark milk as done",
        }
        steps = oracle.solve(info, env)
        assert len(steps) >= 2
        # First step should be find
        assert "find(" in steps[0][0]
        # Should find the task
        assert "#1" in steps[0][1] or "milk" in steps[0][1]


# ═══════════════════════════════════════════════════════════════════════════════
# LangChain / LangGraph Tests
# ═══════════════════════════════════════════════════════════════════════════════

# Skip all LangChain tests if dependencies not installed
try:
    import langchain_core
    import langgraph
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

langchain_skip = pytest.mark.skipif(
    not HAS_LANGCHAIN,
    reason="langchain/langgraph not installed"
)


@langchain_skip
class TestLangChainTools:
    """Tests for kanha.lam.langchain.tools."""

    def test_build_task_tools(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        tools = build_task_tools(store)

        # All 12 actions should be present
        expected = {"add", "list", "find", "count", "edit", "done", "done_all",
                    "delete", "delete_done", "undo", "ask", "finish"}
        assert set(tools.keys()) == expected

    def test_execute_tool_add(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "add", {"title": "buy milk", "due": "today"})
        assert "buy milk" in result
        assert store.count() == 1

    def test_execute_tool_find(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        store.add("buy milk")
        store.add("call mom")
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "find", {"q": "milk"})
        assert "milk" in result
        assert "mom" not in result

    def test_execute_tool_done_with_string_id(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        store.add("test task")
        tools = build_task_tools(store)

        # Pass id as string (as the parser produces)
        result = execute_tool_by_name(tools, "done", {"id": "1"})
        assert "Done" in result or "done" in result.lower()

    def test_execute_tool_unknown_action(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "fly", {"dest": "moon"})
        assert "error" in result.lower()

    def test_execute_tool_finish(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "finish", {"msg": "All done!"})
        assert result == "All done!"

    def test_execute_tool_ask(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "ask", {"msg": "Which one?"})
        assert result == "Which one?"

    def test_execute_tool_count(self, tmp_path):
        from kanha.lam.langchain.tools import build_task_tools, execute_tool_by_name
        path = os.path.join(str(tmp_path), "t.json")
        store = TaskStore(path=path)
        store.add("task1")
        store.add("task2")
        tools = build_task_tools(store)

        result = execute_tool_by_name(tools, "count", {})
        assert "2" in result


@langchain_skip
class TestKanhaLLM:
    """Tests for kanha.lam.langchain.model.KanhaLLM."""

    def test_clean_action(self):
        from kanha.lam.langchain.model import KanhaLLM

        assert KanhaLLM._clean_action('add(title="milk")\n### Observation\nok') == 'add(title="milk")'
        assert KanhaLLM._clean_action("") == ""
        assert KanhaLLM._clean_action("   ") == ""
        assert KanhaLLM._clean_action('done(id=3)\n\nmore text') == "done(id=3)"

    def test_identifying_params(self):
        """KanhaLLM should expose identifying params."""
        from kanha.lam.langchain.model import KanhaLLM
        # Can't instantiate without a real model, but verify the class exists
        assert hasattr(KanhaLLM, '_identifying_params')
        assert hasattr(KanhaLLM, '_llm_type')


@langchain_skip
class TestLangGraphAgent:
    """Tests for kanha.lam.langchain.graph state machine."""

    def test_agent_state_type(self):
        from kanha.lam.langchain.graph import AgentState
        # AgentState should be a TypedDict with the right keys
        assert "goal" in AgentState.__annotations__
        assert "history" in AgentState.__annotations__
        assert "step_count" in AgentState.__annotations__
        assert "is_terminal" in AgentState.__annotations__

    def test_langgraph_result_compatible(self):
        """LangGraphAgentResult should have same fields as AgentResult."""
        from kanha.lam.langchain.graph import LangGraphAgentResult
        from kanha.lam.agent import AgentResult

        lg_fields = {f.name for f in dataclasses.fields(LangGraphAgentResult)}
        base_fields = {f.name for f in dataclasses.fields(AgentResult)}

        # LangGraphAgentResult should have at least all the fields of AgentResult
        assert base_fields.issubset(lg_fields), (
            f"Missing fields in LangGraphAgentResult: {base_fields - lg_fields}"
        )

    def test_after_parse_routing(self):
        """Test the _after_parse conditional edge function."""
        from kanha.lam.langchain.graph import _after_parse

        # Successful parse → execute
        state = {"parsed_name": "add", "parsed_args": {}, "retry_count": 0, "max_retries": 3}
        assert _after_parse(state) == "execute"

        # Failed parse, retries remaining → retry
        state = {"parsed_name": "", "parsed_args": {}, "retry_count": 1, "max_retries": 3}
        assert _after_parse(state) == "retry"

        # Failed parse, retries exhausted → abort
        state = {"parsed_name": "", "parsed_args": {}, "retry_count": 3, "max_retries": 3}
        assert _after_parse(state) == "abort"

    def test_after_execute_routing(self):
        """Test the _after_execute conditional edge function."""
        from kanha.lam.langchain.graph import _after_execute

        # Terminal action → end
        state = {"is_terminal": True, "step_count": 1, "max_steps": 8, "history": []}
        assert _after_execute(state) == "end"

        # Step limit → abort
        state = {"is_terminal": False, "step_count": 8, "max_steps": 8, "history": []}
        assert _after_execute(state) == "abort_limit"

        # Loop detected → abort
        state = {
            "is_terminal": False, "step_count": 3, "max_steps": 8,
            "history": [
                {"action": "done(id=1)"}, {"action": "done(id=1)"}, {"action": "done(id=1)"}
            ]
        }
        assert _after_execute(state) == "abort_loop"

        # Normal continue
        state = {
            "is_terminal": False, "step_count": 1, "max_steps": 8,
            "history": [{"action": "add(title='x')"}]
        }
        assert _after_execute(state) == "continue"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
