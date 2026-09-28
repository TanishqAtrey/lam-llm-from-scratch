"""
kanha/lam/agent_cli.py
Interactive terminal interface for Kanha Tasks LAM.

Supports both the base TaskAgent and the LangGraph-backed agent.
Select with --langgraph flag.
"""

import sys
from rich.console import Console
from rich.panel import Panel

from kanha.core.model import KanhaModel
from kanha.core.tokenizer import KanhaTokenizer
from kanha.lam.environment import TaskEnvironment


def run_agent_cli(args):
    """REPL for Kanha Tasks."""
    console = Console()

    console.print(f"Loading model from {args.model}...")
    model = KanhaModel.from_pretrained(args.model)
    tokenizer = KanhaTokenizer()

    env = TaskEnvironment(task_store_path=args.tasks)

    # Choose agent backend
    use_langgraph = getattr(args, "langgraph", False)

    if use_langgraph:
        try:
            from kanha.lam.langchain.graph import LangGraphTaskAgent
            agent = LangGraphTaskAgent(model, tokenizer, env)
            backend_name = "LangGraph"
        except ImportError as e:
            console.print(f"[yellow]LangGraph not available ({e}), falling back to base agent.[/yellow]")
            from kanha.lam.agent import TaskAgent
            agent = TaskAgent(model, tokenizer, env)
            backend_name = "Base"
    else:
        from kanha.lam.agent import TaskAgent
        agent = TaskAgent(model, tokenizer, env)
        backend_name = "Base"

    console.print(Panel(
        f"[bold green]Kanha Tasks LAM Agent CLI[/bold green]\n"
        f"Backend: [cyan]{backend_name}[/cyan]\n"
        f"Commands: /tasks, /reset, /undo, /help, /exit",
        title="Welcome"
    ))

    while True:
        try:
            goal = console.input("[bold blue]You: [/bold blue]")
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        goal = goal.strip()
        if not goal:
            continue

        if goal == "/exit":
            break
        elif goal == "/help":
            console.print(
                "Commands:\n"
                "  /tasks - show current tasks\n"
                "  /reset - clear tasks\n"
                "  /undo  - undo last action\n"
                "  /help  - show this help\n"
                "  /exit  - quit"
            )
            continue
        elif goal == "/tasks":
            tasks, remaining = env.store.list(limit=100)
            if not tasks:
                console.print("No tasks.")
            else:
                for t in tasks:
                    status_icon = "✓" if t.status == "done" else "○"
                    console.print(
                        f"  {status_icon} #{t.id} {t.title} | "
                        f"{t.due or '-'} | {t.priority} | "
                        f"{t.tag or '-'} | {t.status}"
                    )
                if remaining > 0:
                    console.print(f"  (+{remaining} more)")
            continue
        elif goal == "/reset":
            env.reset()
            env.store.save()
            console.print("[green]Tasks cleared.[/green]")
            continue
        elif goal == "/undo":
            try:
                msg = env.store.undo()
                console.print(f"[green]{msg}[/green]")
            except ValueError as e:
                console.print(f"[red]{e}[/red]")
            continue

        # Run agent
        result = agent.run(goal)
        for i, step in enumerate(result.steps):
            console.print(f"  Step {i+1}: [cyan]{step['action']}[/cyan]")
            console.print(f"    [dim]→ {step['observation']}[/dim]")

        if result.success:
            console.print(f"[bold green]✓ {result.final_message}[/bold green]")
        elif result.aborted:
            console.print(f"[bold red]✗ Aborted: {result.abort_reason}[/bold red]")
        else:
            console.print("[bold red]✗ Failed to achieve goal.[/bold red]")

        # Save after interaction
        env.store.save()
