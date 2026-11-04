"""
kanha/lam/data/oracle.py
Programmatic solver that produces gold-standard action sequences.

Each solver method takes template_info and a TaskEnvironment,
executes actions through env.step(), and returns the (action, observation)
pairs as recorded by the environment.
"""

from __future__ import annotations
import re


class Oracle:
    def __init__(self):
        pass

    def solve(self, template_info: dict, env) -> list[tuple[str, str]]:
        """Dispatch to per-tier solver based on template_info['solver']."""
        solver_name = template_info.get("solver")
        if not solver_name or not hasattr(self, solver_name):
            return []
        solver = getattr(self, solver_name)
        return solver(template_info, env)

    # ── Tier 1: Direct add ────────────────────────────────────────────────

    def _solve_direct_add(self, info, env):
        slots = info.get("slot_values", {})

        parts = []
        if "title" in slots:
            parts.append(f'title="{slots["title"]}"')
        if "due_phrase_val" in info:
            parts.append(f"due={info['due_phrase_val']}")
        elif "due_phrase" in slots:
            parts.append(f"due={slots['due_phrase']}")
        if "prio_phrase_val" in info:
            parts.append(f"prio={info['prio_phrase_val']}")
        elif "prio_phrase" in slots:
            parts.append(f"prio={slots['prio_phrase']}")
        if "tag" in slots:
            parts.append(f"tag={slots['tag']}")

        action = f"add({', '.join(parts)})"
        obs, _ = env.step(action)

        finish = 'finish(msg="Task added.")'
        obs_f, _ = env.step(finish)
        return [(action, obs), (finish, obs_f)]

    # ── Tier 2: Query count ───────────────────────────────────────────────

    def _solve_query_count(self, info, env):
        slots = info.get("slot_values", {})
        parts = []

        if "due_phrase_val" in info:
            parts.append(f"due={info['due_phrase_val']}")
        if "prio_phrase_val" in info:
            parts.append(f"prio={info['prio_phrase_val']}")
        if "tag" in slots:
            parts.append(f"tag={slots['tag']}")

        action = f"count({', '.join(parts)})"
        obs, _ = env.step(action)

        # Parse the count from "Count: N"
        count = 0
        m = re.search(r"Count:\s*(\d+)", obs)
        if m:
            count = int(m.group(1))

        info["expected_reply"] = {"contains_number": count}
        finish = f'finish(msg="You have {count} matching tasks.")'
        obs_f, _ = env.step(finish)
        return [(action, obs), (finish, obs_f)]

    # ── Tier 2: Query list ────────────────────────────────────────────────

    def _solve_query_list(self, info, env):
        slots = info.get("slot_values", {})
        parts = []

        if "due_phrase_val" in info:
            parts.append(f"due={info['due_phrase_val']}")
        if "prio_phrase_val" in info:
            parts.append(f"prio={info['prio_phrase_val']}")
        if "tag" in slots:
            parts.append(f"tag={slots['tag']}")

        action = f"list({', '.join(parts)})"
        obs, _ = env.step(action)

        info["expected_reply"] = {"contains_phrase": "tasks"}
        finish = 'finish(msg="Here are your tasks.")'
        obs_f, _ = env.step(finish)
        return [(action, obs), (finish, obs_f)]

    # ── Tier 3: Lookup then act ───────────────────────────────────────────

    def _solve_lookup_then_act(self, info, env):
        slots = info.get("slot_values", {})
        query = slots.get("query", "")

        # Step 1: find
        act1 = f'find(q="{query}")'
        obs1, _ = env.step(act1)

        # Extract task ID
        target_id = info.get("target_id")
        task_id = None
        if target_id is not None and f"#{target_id} " in obs1:
            task_id = str(target_id)
        else:
            target_title = info.get("query_val", query)
            if target_title:
                m = re.search(rf"#(\d+)\s+.*{re.escape(target_title)}", obs1, re.IGNORECASE)
                if m:
                    task_id = m.group(1)
        if not task_id:
            m = re.search(r"#(\d+)", obs1)
            if m:
                task_id = m.group(1)

        if not task_id:
            finish = 'finish(msg="Could not find the task.")'
            obs_f, _ = env.step(finish)
            return [(act1, obs1), (finish, obs_f)]
        goal = (info.get("goal") or info.get("goal_text", "")).lower()
        raw_tmpl = (info.get("goal") or "").lower() if "{" in info.get("goal", "") else ""

        # Step 2: determine the action based on goal intent without substring collisions
        if raw_tmpl:
            if any(k in raw_tmpl for k in ["done", "complete", "finish", "mark"]):
                act2 = f"done(id={task_id})"
            elif any(k in raw_tmpl for k in ["delete", "remove"]):
                act2 = f"delete(id={task_id})"
            elif "rename" in raw_tmpl or "new_title" in slots:
                new_title = slots.get("new_title", "new task")
                act2 = f'edit(id={task_id}, field=title, value="{new_title}")'
            elif "tag" in raw_tmpl:
                tag = slots.get("tag", "home")
                act2 = f'edit(id={task_id}, field=tag, value={tag})'
            elif bool(re.search(r"\bmove\b", raw_tmpl)) or "reschedule" in raw_tmpl:
                due = info.get("due_phrase_val", "today")
                act2 = f'edit(id={task_id}, field=due, value={due})'
            elif "prio" in raw_tmpl or "change" in raw_tmpl or "set" in raw_tmpl:
                prio = info.get("prio_val_val", info.get("prio_phrase_val", "med"))
                act2 = f'edit(id={task_id}, field=priority, value={prio})'
            else:
                act2 = f"done(id={task_id})"
        elif "new_title" in slots or "rename" in goal:
            new_title = slots.get("new_title", "new task")
            act2 = f'edit(id={task_id}, field=title, value="{new_title}")'
        elif "tag" in slots and ("tag" in goal and "as" in goal):
            tag = slots.get("tag", "home")
            act2 = f'edit(id={task_id}, field=tag, value={tag})'
        elif any(kw in goal for kw in ["mark", "complete", "done"]):
            act2 = f"done(id={task_id})"
        elif "delete" in goal or "remove" in goal:
            act2 = f"delete(id={task_id})"
        elif bool(re.search(r"\bmove\b", goal)) or "reschedule" in goal:
            due = info.get("due_phrase_val", "today")
            act2 = f'edit(id={task_id}, field=due, value={due})'
        elif "prio_phrase_val" in info or "prio_val_val" in info or "priority" in goal or (("change" in goal or "set" in goal) and any(p in goal for p in ["low", "med", "high", "prio"])):
            prio = info.get("prio_val_val", info.get("prio_phrase_val", "med"))
            act2 = f'edit(id={task_id}, field=priority, value={prio})'
        else:
            act2 = f"done(id={task_id})"

        obs2, _ = env.step(act2)

        finish = 'finish(msg="Task updated.")'
        obs_f, _ = env.step(finish)
        return [(act1, obs1), (act2, obs2), (finish, obs_f)]

    # ── Tier 4: Bulk ──────────────────────────────────────────────────────

    def _solve_bulk(self, info, env):
        slots = info.get("slot_values", {})
        parts = []

        if "due_phrase_val" in info:
            parts.append(f"due={info['due_phrase_val']}")
        if "prio_phrase_val" in info:
            parts.append(f"prio={info['prio_phrase_val']}")
        if "tag" in slots:
            parts.append(f"tag={slots['tag']}")

        goal = (info.get("goal") or info.get("goal_text", "")).lower()
        if "delete" in goal and ("completed" in goal or "done" in goal):
            action = "delete_done()"
        else:
            action = f"done_all({', '.join(parts)})"

        obs, _ = env.step(action)

        finish = 'finish(msg="Bulk action completed.")'
        obs_f, _ = env.step(finish)
        return [(action, obs), (finish, obs_f)]

    # ── Tier 4: Compound ──────────────────────────────────────────────────

    def _solve_compound(self, info, env):
        slots = info.get("slot_values", {})

        t1 = slots.get("title1", "task one")
        d1 = info.get("due1_val", info.get("due_phrase_val", "today"))
        t2 = slots.get("title2", "task two")
        d2 = info.get("due2_val", info.get("due_phrase_val", "tomorrow"))

        act1 = f'add(title="{t1}", due={d1})'
        obs1, _ = env.step(act1)

        act2 = f'add(title="{t2}", due={d2})'
        obs2, _ = env.step(act2)

        finish = 'finish(msg="Both tasks added.")'
        obs_f, _ = env.step(finish)
        return [(act1, obs1), (act2, obs2), (finish, obs_f)]

    # ── Tier 5: Narrowing ─────────────────────────────────────────────────

    def _solve_narrowing(self, info, env):
        act1 = "list()"
        obs1, _ = env.step(act1)

        info["expected_reply"] = {"contains_phrase": "tasks"}
        # Check if there are more tasks than shown
        if "(+" in obs1:
            act2 = "list(due=today)"
            obs2, _ = env.step(act2)
            if "No tasks found" in obs2:
                finish = 'finish(msg="You have no tasks due today.")'
            else:
                finish = 'finish(msg="Here are your most pressing tasks.")'
            obs_f, _ = env.step(finish)
            return [(act1, obs1), (act2, obs2), (finish, obs_f)]
        else:
            finish = 'finish(msg="Here are your tasks.")'
            obs_f, _ = env.step(finish)
            return [(act1, obs1), (finish, obs_f)]

    # ── Tier 6: Error recovery ────────────────────────────────────────────

    def _solve_error_recovery(self, info, env):
        slots = info.get("slot_values", {})
        goal = (info.get("goal") or info.get("goal_text", "")).lower()

        # Goals with explicit bad IDs (e.g. task 99, 88, 77, 999) - only if no query slot
        if "query" not in slots:
            m_id = re.search(r"\b(?:task\s+|#|of\s+)?(\d+)\b", goal)
            if m_id:
                tid = m_id.group(1)
                if int(tid) > 50:  # likely a bad ID
                    if "done" in goal or "complete" in goal or "mark" in goal:
                        act1 = f"done(id={tid})"
                    elif "delete" in goal or "remove" in goal:
                        act1 = f"delete(id={tid})"
                    elif "priority" in goal or "prio" in goal:
                        act1 = f"edit(id={tid}, field=priority, value=high)"
                    elif "tag" in goal:
                        act1 = f"edit(id={tid}, field=tag, value=work)"
                    else:
                        act1 = f"edit(id={tid}, field=title, value=new)"
                    obs1, _ = env.step(act1)
                    finish = 'finish(msg="Could not find that task.")'
                    obs_f, _ = env.step(finish)
                    return [(act1, obs1), (finish, obs_f)]

        # Goals with query that won't match
        query = slots.get("query", "missing_task_xyz")
        act1 = f'find(q="{query}")'
        obs1, _ = env.step(act1)

        finish = 'finish(msg="Could not find matching task.")'
        obs_f, _ = env.step(finish)
        return [(act1, obs1), (finish, obs_f)]

    # ── Tier 7: Ambiguity ─────────────────────────────────────────────────

    def _solve_ambiguity(self, info, env):
        slots = info.get("slot_values", {})
        query = slots.get("ambiguous_query", "task")

        act1 = f'find(q="{query}")'
        obs1, _ = env.step(act1)

        info["expected_reply"] = {"contains_phrase": "which"}
        ask = 'ask(msg="Which one do you mean?")'
        obs_a, _ = env.step(ask)
        return [(act1, obs1), (ask, obs_a)]

    # ── Tier 8: Out of scope ──────────────────────────────────────────────

    def _solve_out_of_scope(self, info, env):
        info["expected_reply"] = {"contains_phrase": "sorry"}
        finish = 'finish(msg="Sorry, I can only manage tasks.")'
        obs, _ = env.step(finish)
        return [(finish, obs)]
