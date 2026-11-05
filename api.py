"""
api.py
FastAPI REST server for KANHA.

Endpoints:
    POST /chat      — single-turn chat
    POST /reset     — clear memory
    GET  /health    — health check

Run:
    python main.py api --model models/finetuned/sft_final.pt --port 8000
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List

from kanha.inference.engine import InferenceEngine
from kanha.utils.logging import get_logger

log = get_logger(__name__)

# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: Optional[str] = None
    prompt: Optional[str] = None
    stream: Optional[bool] = False
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    model: str = "kanha"


class AgentRequest(BaseModel):
    goal: str
    tasks_path: Optional[str] = "data/lam/tasks.json"
    use_langgraph: Optional[bool] = False


class AgentStepResponse(BaseModel):
    step: int
    action: str
    observation: str


class AgentResponse(BaseModel):
    goal: str
    steps: List[AgentStepResponse]
    final_message: Optional[str] = None
    success: bool
    aborted: bool
    abort_reason: Optional[str] = None


# ── App factory ───────────────────────────────────────────────────────────────

def _resolve_model_path(model_path: Optional[str] = None) -> str:
    """Resolves the model checkpoint path from argument, env vars, or config."""
    if model_path:
        return model_path
    env_path = os.getenv("KANHA_MODEL") or os.getenv("KANHA_MODEL_PATH")
    if env_path:
        return env_path
    try:
        from kanha.utils.config import cfg
        default_dir = getattr(cfg.paths, "finetuned_model", "models/finetuned/")
        for cand_name in ["dpo_final.pt", "dpo_best.pt", "sft_final.pt", "sft_best.pt"]:
            candidate = os.path.join(default_dir, cand_name) if default_dir.endswith("/") else default_dir
            if os.path.exists(candidate):
                return candidate
    except Exception:
        pass
    for fallback in [
        "models/finetuned/dpo_final.pt",
        "models/finetuned/dpo_best.pt",
        "models/finetuned/sft_final.pt",
        "models/finetuned/sft_best.pt",
        "models/base/final_model.pt",
        "models/lam/dpo_final.pt",
        "models/lam/dpo_best.pt",
        "models/lam/lam_final.pt",
    ]:
        if os.path.exists(fallback):
            return fallback
    return "models/finetuned/sft_final.pt"


def create_app(
    model_path: Optional[str] = None,
    lam_model_path: Optional[str] = None,
    index_dir: Optional[str] = None,
) -> FastAPI:
    resolved_path = _resolve_model_path(model_path)

    app = FastAPI(
        title="KANHA AI API",
        description="Knowledge-Augmented Neural Heuristic Assistant",
        version="0.1.0",
    )

    # Strictly lazy engine loading to prevent double-loading at import time
    _engine: Optional[InferenceEngine] = None
    _lam_model = None
    _sessions: dict = {}

    def get_engine() -> InferenceEngine:
        nonlocal _engine
        if _engine is None:
            if not os.path.exists(resolved_path):
                raise HTTPException(
                    status_code=503,
                    detail=f"Model checkpoint not found at '{resolved_path}'. Set KANHA_MODEL environment variable or train a model first.",
                )
            _engine = InferenceEngine.from_pretrained(
                model_path=resolved_path,
                index_dir=index_dir,
            )
        return _engine

    def get_lam_model(base_engine: InferenceEngine):
        nonlocal _lam_model
        if _lam_model is None:
            lam_path = lam_model_path or os.getenv("KANHA_LAM_MODEL")
            if not lam_path:
                for candidate in [
                    "models/lam/dpo_final.pt",
                    "models/lam/dpo_best.pt",
                    "models/lam/lam_final.pt",
                    "models/lam/lam_sft.pt",
                ]:
                    if os.path.exists(candidate):
                        lam_path = candidate
                        break
            if lam_path and os.path.exists(lam_path):
                try:
                    from kanha.core.model import KanhaModel
                    _lam_model = KanhaModel.from_pretrained(lam_path)
                    log.info(f"Loaded dedicated LAM agent model from {lam_path}")
                except Exception as e:
                    log.warning(f"Could not load LAM model from {lam_path}: {e}")
                    _lam_model = base_engine.model
            else:
                log.warning("No dedicated LAM model checkpoint found; falling back to base/sft model.")
                _lam_model = base_engine.model
        return _lam_model

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": resolved_path,
            "loaded": _engine is not None,
        }

    @app.post("/chat")
    def chat(req: ChatRequest):
        text = req.prompt if req.prompt is not None else req.message
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Message or prompt cannot be empty.")
        try:
            engine = get_engine()
            # Session-isolated memory handling
            from kanha.memory.short_term import ShortTermMemory
            from kanha.utils.config import cfg
            if req.session_id:
                if req.session_id in _sessions:
                    sess_mem = _sessions.pop(req.session_id)
                    _sessions[req.session_id] = sess_mem
                else:
                    if len(_sessions) >= 1000:
                        oldest_key = next(iter(_sessions))
                        del _sessions[oldest_key]
                    sess_mem = ShortTermMemory(max_turns=cfg.memory.short_term_limit)
                    _sessions[req.session_id] = sess_mem
            else:
                # Stateless request: use ephemeral isolated memory so un-sessioned requests
                # do not leak or share conversation context with other users or engine.memory
                sess_mem = ShortTermMemory(max_turns=0)

            if req.stream:
                return StreamingResponse(engine.chat_stream(text, memory=sess_mem), media_type="text/plain")
            response = engine.chat(text, stream=False, memory=sess_mem)
            return ChatResponse(response=response)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    def reset(session_id: Optional[str] = None):
        try:
            if session_id and session_id in _sessions:
                _sessions[session_id].clear()
                return {"status": f"session {session_id} memory cleared"}
            engine = get_engine()
            engine.reset_memory()
            return {"status": "memory cleared"}
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Reset error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/agent", response_model=AgentResponse)
    def agent_endpoint(req: AgentRequest):
        """Runs a LAM agent trajectory for the given goal."""
        if not req.goal.strip():
            raise HTTPException(status_code=400, detail="Goal cannot be empty.")
        try:
            engine = get_engine()
            lam_model = get_lam_model(engine)
            from kanha.lam.environment import TaskEnvironment

            # Path traversal prevention: enforce sandbox path strictly inside data/lam/
            import datetime
            safe_base = os.path.abspath("data/lam")
            os.makedirs(safe_base, exist_ok=True)
            candidate = os.path.abspath(os.path.join(safe_base, req.tasks_path or "tasks.json"))
            try:
                if os.path.commonpath([safe_base, candidate]) != safe_base:
                    safe_tasks_path = os.path.join(safe_base, os.path.basename(req.tasks_path or "tasks.json"))
                else:
                    safe_tasks_path = candidate
            except ValueError:
                safe_tasks_path = os.path.join(safe_base, os.path.basename(req.tasks_path or "tasks.json"))

            env = TaskEnvironment(task_store_path=safe_tasks_path, reference_date=datetime.date.today().isoformat())

            if req.use_langgraph:
                try:
                    from kanha.lam.langchain.graph import LangGraphTaskAgent
                    agent = LangGraphTaskAgent(
                        model=lam_model,
                        tokenizer=engine.tokenizer,
                        environment=env,
                    )
                except ImportError:
                    from kanha.lam.agent import TaskAgent
                    agent = TaskAgent(
                        model=lam_model,
                        tokenizer=engine.tokenizer,
                        environment=env,
                    )
            else:
                from kanha.lam.agent import TaskAgent
                agent = TaskAgent(
                    model=lam_model,
                    tokenizer=engine.tokenizer,
                    environment=env,
                )

            result = agent.run(req.goal)
            env.store.save()

            return AgentResponse(
                goal=result.goal,
                steps=[
                    AgentStepResponse(
                        step=i + 1,
                        action=s["action"],
                        observation=s["observation"],
                    )
                    for i, s in enumerate(result.steps)
                ],
                final_message=result.final_message,
                success=result.success,
                aborted=result.aborted,
                abort_reason=result.abort_reason,
            )
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Agent error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app


# ── Top-level application instance for `uvicorn api:app` ─────────────────────
app = create_app()