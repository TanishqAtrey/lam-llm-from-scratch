"""
main.py
KANHA entry point.

Modes:
    python main.py chat    --model models/finetuned/sft_final.pt
    python main.py train   --data  data/processed/train.npy
    python main.py finetune --base models/base/final_model.pt --data data/processed/sft.jsonl
    python main.py index   --docs  data/raw/
    python main.py api     --model models/finetuned/sft_final.pt
"""

import argparse
import dataclasses
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="kanha",
        description="KANHA AI — Knowledge-Augmented Neural Heuristic Assistant",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── chat ──────────────────────────────────────────────────────────────
    chat_p = sub.add_parser("chat", help="Start terminal chat")
    chat_p.add_argument("--model",  required=True)
    chat_p.add_argument("--index",  default=None)
    chat_p.add_argument("--stream", action="store_true")
    chat_p.add_argument("--tools",  action="store_true")

    # ── train ─────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", aliases=["pretrain"], help="Pre-train from scratch")
    train_p.add_argument("--data",   required=True)
    train_p.add_argument("--resume", default=None)
    train_p.add_argument("--val_fraction", type=float, default=0.1,
                         help="Fraction of data for validation (default 0.1)")
    train_p.add_argument("--patience", type=int, default=5,
                         help="Early stop after N evals without val improvement")
    train_p.add_argument("--eval_every", type=int, default=None,
                         help="Evaluate val loss every N steps (default=save_every)")

    # ── finetune ──────────────────────────────────────────────────────────
    ft_p = sub.add_parser("finetune", aliases=["sft"], help="SFT fine-tuning")
    ft_p.add_argument("--base_model", "--base", "--model", dest="base_model", required=True)
    ft_p.add_argument("--data",   required=True)
    ft_p.add_argument("--output", default="models/finetuned/")
    ft_p.add_argument("--epochs", type=int, default=3)
    ft_p.add_argument("--batch_size", type=int, default=1)
    ft_p.add_argument("--lr",         type=float, default=3e-5)
    ft_p.add_argument("--val_fraction", type=float, default=0.1,
                       help="Fraction of data for validation (default 0.1)")
    ft_p.add_argument("--patience", type=int, default=3,
                       help="Early stop after N epochs without val improvement")

    # ── train-tokenizer ───────────────────────────────────────────────────
    tok_p = sub.add_parser("train-tokenizer", help="Train a new BPE tokenizer")
    tok_p.add_argument("--data", "--input", dest="data", required=True, help="Input text file(s)")
    tok_p.add_argument("--output", "--model_prefix", dest="output", default="models/tokenizer/tokenizer", help="Output prefix")
    tok_p.add_argument("--vocab_size", type=int, default=None, help="Vocabulary size")

    # ── index ─────────────────────────────────────────────────────────────
    idx_p = sub.add_parser("index", help="Build FAISS index from documents")
    idx_p.add_argument("--docs",   required=True, help="Directory of .txt files")
    idx_p.add_argument("--output", default="data/embeddings/")

    # ── dpo ────────────────────────────────────────────────────────────────
    dpo_p = sub.add_parser("dpo", help="DPO alignment training")
    dpo_p.add_argument("--base_model", required=True)
    dpo_p.add_argument("--data",       required=True)
    dpo_p.add_argument("--output",     default="models/finetuned/")
    dpo_p.add_argument("--epochs",     type=int, default=1)
    dpo_p.add_argument("--batch_size", type=int, default=1)
    dpo_p.add_argument("--lr",         type=float, default=1e-6)
    dpo_p.add_argument("--beta",       type=float, default=0.1)
    dpo_p.add_argument("--lam_mode",   action="store_true", help="Enable LAM DPO mode for agent trajectories")

    # ── api ───────────────────────────────────────────────────────────────
    api_p = sub.add_parser("api", help="Start FastAPI server")
    api_p.add_argument("--model", required=True)
    api_p.add_argument("--host",  default="0.0.0.0")
    api_p.add_argument("--port",  type=int, default=8000)

    # ── lam-generate ──────────────────────────────────────────────────────
    lamgen_p = sub.add_parser("lam-generate", help="Generate LAM training data")
    lamgen_p.add_argument("--output", default="data/lam/trajectories.jsonl")
    lamgen_p.add_argument("--count", "--num", dest="count", type=int, default=100000)
    lamgen_p.add_argument("--seed",   type=int, default=42)

    # ── lam-train ─────────────────────────────────────────────────────────
    lamtrain_p = sub.add_parser("lam-train", help="SFT training for LAM")
    lamtrain_p.add_argument("--base_model", "--model", dest="base_model", required=True)
    lamtrain_p.add_argument("--data",       required=True)
    lamtrain_p.add_argument("--output",     default="models/lam/")
    lamtrain_p.add_argument("--epochs",     type=int, default=3)
    lamtrain_p.add_argument("--batch_size", type=int, default=4)
    lamtrain_p.add_argument("--lr",         type=float, default=3e-5)
    lamtrain_p.add_argument("--val_fraction", type=float, default=0.1,
                             help="Fraction of data for validation (default 0.1)")
    lamtrain_p.add_argument("--patience", type=int, default=3,
                             help="Early stop after N epochs without val improvement")

    # ── lam-dpo ───────────────────────────────────────────────────────────
    lamdpo_p = sub.add_parser("lam-dpo", help="DPO alignment for LAM")
    lamdpo_p.add_argument("--base_model", "--model", dest="base_model", required=True)
    lamdpo_p.add_argument("--data",       required=True)
    lamdpo_p.add_argument("--output",     default="models/lam/")
    lamdpo_p.add_argument("--epochs",     type=int, default=1)
    lamdpo_p.add_argument("--batch_size", type=int, default=1)
    lamdpo_p.add_argument("--lr",         type=float, default=1e-6)
    lamdpo_p.add_argument("--beta",       type=float, default=0.1)

    # ── agent ─────────────────────────────────────────────────────────────
    agent_p = sub.add_parser("agent", help="Run Kanha Tasks agent (LAM)")
    agent_p.add_argument("--model", required=True)
    agent_p.add_argument("--tasks", default="data/lam/tasks.json")
    agent_p.add_argument("--langgraph", action="store_true",
                         help="Use LangGraph-backed agent (requires langchain, langgraph)")

    # ── lam-eval ──────────────────────────────────────────────────────────
    lameval_p = sub.add_parser("lam-eval", help="Evaluate LAM agent")
    lameval_p.add_argument("--model",     required=True)
    lameval_p.add_argument("--test_data", "--data", dest="test_data", required=True)
    lameval_p.add_argument("--output",    default="results/lam_eval.json")

    # ── mine-dpo ──────────────────────────────────────────────────────────
    mine_p = sub.add_parser("mine-dpo", help="Auto-mine DPO pairs from agent trajectories")
    mine_p.add_argument("--model",     required=True)
    mine_p.add_argument("--eval_data", required=True,
                        help="JSONL with goals, worlds, and oracle steps")
    mine_p.add_argument("--output",    default="data/lam/dpo_pairs.jsonl")
    mine_p.add_argument("--max_goals", type=int, default=500,
                        help="Max goals to evaluate (default 500)")
    mine_p.add_argument("--reference_date", default="2025-01-15")

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────
    if args.command == "chat":
        from cli import run_cli
        run_cli(args)

    elif args.command in ("train", "pretrain"):
        from kanha.training.train import train
        train(args)

    elif args.command in ("finetune", "sft"):
        from kanha.finetune.sft_train import sft_train
        sft_train(args)

    elif args.command == "train-tokenizer":
        from kanha.core.tokenizer import train_tokenizer
        train_tokenizer(
            input_files=args.data,
            model_prefix=args.output,
            vocab_size=args.vocab_size,
        )

    elif args.command == "dpo":
        from kanha.finetune.dpo_train import dpo_train
        dpo_train(args)

    elif args.command == "index":
        _build_index(args)

    elif args.command == "api":
        _start_api(args)

    elif args.command == "lam-generate":
        _lam_generate(args)

    elif args.command == "lam-train":
        from kanha.finetune.lam_train import lam_sft_train
        lam_sft_train(args)

    elif args.command == "lam-dpo":
        from kanha.finetune.dpo_train import dpo_train
        args.lam_mode = True
        dpo_train(args)

    elif args.command == "agent":
        _run_agent(args)

    elif args.command == "lam-eval":
        _lam_eval(args)

    elif args.command == "mine-dpo":
        from scripts.mine_dpo_pairs import mine_dpo_pairs
        mine_dpo_pairs(args)


def _build_index(args):
    """Builds a FAISS index from a directory of .txt files."""
    import os
    from kanha.rag.chunker import Chunker
    from kanha.rag.retriever import Retriever
    from kanha.utils.logging import get_logger

    log = get_logger("index")

    docs = []
    for fname in os.listdir(args.docs):
        if fname.endswith(".txt"):
            fpath = os.path.join(args.docs, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                docs.append({"id": fname, "text": f.read()})

    log.info(f"Found {len(docs)} documents in {args.docs}")

    chunker   = Chunker()
    chunks    = chunker.chunk_documents(docs)
    retriever = Retriever()
    retriever.build_index_from_chunks(chunks, save_dir=args.output)
    log.info(f"Index saved to {args.output}")


def _start_api(args):
    """Starts FastAPI server."""
    try:
        import uvicorn
        from api import create_app
        app = create_app(model_path=args.model)
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("FastAPI/uvicorn not installed. Run: pip install fastapi uvicorn")


def _lam_generate(args):
    """Generates LAM training data."""
    from kanha.lam.data.generator import TrajectoryGenerator
    from kanha.utils.helpers import ensure_dir

    ensure_dir(os.path.dirname(args.output))
    gen = TrajectoryGenerator(
        num_trajectories=args.count,
        seed=args.seed,
    )
    stats = gen.generate(args.output)
    print(f"\nGenerated {stats['total']:,} trajectories to {args.output}")
    for tier, count in sorted(stats.get("per_tier", {}).items()):
        print(f"  {tier}: {count:,}")


def _run_agent(args):
    """Runs the Kanha Tasks agent REPL."""
    from kanha.lam.agent_cli import run_agent_cli
    run_agent_cli(args)


def _lam_eval(args):
    """Evaluates the LAM agent on a test set."""
    import json as _json
    from kanha.core.model import KanhaModel
    from kanha.core.tokenizer import KanhaTokenizer
    from kanha.lam.evaluator import LAMEvaluator
    from kanha.utils.helpers import ensure_dir

    model = KanhaModel.from_pretrained(args.model)
    tokenizer = KanhaTokenizer()

    evaluator = LAMEvaluator(model, tokenizer, args.test_data)
    report = evaluator.evaluate()

    print(report.summary())

    ensure_dir(os.path.dirname(args.output))
    with open(args.output, "w") as f:
        _json.dump(dataclasses.asdict(report), f, indent=2, default=str)
    print(f"\nDetailed results saved to {args.output}")



if __name__ == "__main__":
    main()