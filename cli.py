"""
cli.py
KANHA terminal chat interface.

Run:
    python cli.py --model models/finetuned/sft_final.pt
    python cli.py --model models/finetuned/sft_final.pt --index data/embeddings/
    python cli.py --model models/finetuned/sft_final.pt --stream
"""

import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from kanha.inference.engine import InferenceEngine
from kanha.utils.logging import print_banner

console = Console()

COMMANDS = {
    "/reset" : "Clear conversation memory",
    "/exit"  : "Quit KANHA",
    "/help"  : "Show this help",
    "/memory": "Show current memory",
}


def print_help():
    console.print(Panel(
        "\n".join(f"[cyan]{cmd}[/cyan]  {desc}" for cmd, desc in COMMANDS.items()),
        title="[bold magenta]Commands[/bold magenta]",
        border_style="magenta",
    ))


def run_cli(args):
    print_banner()

    console.print("[bold cyan]Loading KANHA...[/bold cyan]")

    engine = InferenceEngine.from_pretrained(
        model_path=args.model,
        index_dir=args.index if args.index else None,
        use_rag=bool(args.index),
        use_tools=args.tools,
    )

    console.print("[bold green]KANHA is ready! Type your message below.[/bold green]")
    console.print("[dim]Type /help for commands[/dim]\n")

    while True:
        try:
            # Get user input
            user_input = console.input("[bold yellow]You:[/bold yellow] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold red]Goodbye![/bold red]")
            break

        if not user_input:
            continue

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/exit":
                console.print("[bold red]Goodbye![/bold red]")
                break
            elif cmd == "/reset":
                engine.reset_memory()
                console.print("[green]Memory cleared.[/green]")
            elif cmd == "/help":
                print_help()
            elif cmd == "/memory":
                ctx = engine.memory.get_context("summary")
                console.print(Panel(ctx or "(empty)", title="Memory", border_style="blue"))
            else:
                console.print(f"[red]Unknown command: {cmd}[/red]")
            continue

        # Generate response
        try:
            if args.stream:
                # Streaming mode — tokens printed inside engine
                response = engine.chat(user_input, stream=True)
            else:
                with console.status("[bold cyan]Thinking...[/bold cyan]"):
                    response = engine.chat(user_input, stream=False)
                console.print(Panel(
                    Text(response, style="white"),
                    title="[bold magenta]KANHA[/bold magenta]",
                    border_style="magenta",
                ))
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KANHA Terminal Chat")
    parser.add_argument("--model",  required=True, help="Path to model .pt file")
    parser.add_argument("--index",  default=None,  help="Path to FAISS index directory")
    parser.add_argument("--stream", action="store_true", help="Enable streaming output")
    parser.add_argument("--tools",  action="store_true", help="Enable agent tools")
    args = parser.parse_args()
    run_cli(args)