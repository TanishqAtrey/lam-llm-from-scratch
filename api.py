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
        candidate = os.path.join(default_dir, "sft_final.pt") if default_dir.endswith("/") else default_dir
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    for fallback in [
        "models/finetuned/sft_final.pt",
        "models/base/final_model.pt",
        "models/lam/lam_final.pt",
    ]:
        if os.path.exists(fallback):
            return fallback
    return "models/finetuned/sft_final.pt"


def create_app(model_path: Optional[str] = None, index_dir: Optional[str] = None) -> FastAPI:
    resolved_path = _resolve_model_path(model_path)

    app = FastAPI(
        title="KANHA AI API",
        description="Knowledge-Augmented Neural Heuristic Assistant",
        version="0.1.0",
    )

    # Lazy or immediate engine loading
    _engine: Optional[InferenceEngine] = None
    if os.path.exists(resolved_path):
        try:
            _engine = InferenceEngine.from_pretrained(
                model_path=resolved_path,
                index_dir=index_dir,
            )
        except Exception as e:
            log.warning(f"Deferred engine loading: failed to pre-load from '{resolved_path}': {e}")

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

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "model": resolved_path,
            "loaded": _engine is not None or os.path.exists(resolved_path),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        text = req.prompt if req.prompt is not None else req.message
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Message or prompt cannot be empty.")
        try:
            engine = get_engine()
            response = engine.chat(text, stream=False)
            return ChatResponse(response=response)
        except HTTPException:
            raise
        except Exception as e:
            log.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    def reset():
        try:
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
            from kanha.lam.environment import TaskEnvironment

            env = TaskEnvironment(task_store_path=req.tasks_path)

            if req.use_langgraph:
                try:
                    from kanha.lam.langchain.graph import LangGraphTaskAgent
                    agent = LangGraphTaskAgent(
                        model=engine.model,
                        tokenizer=engine.tokenizer,
                        environment=env,
                    )
                except ImportError:
                    from kanha.lam.agent import TaskAgent
                    agent = TaskAgent(
                        model=engine.model,
                        tokenizer=engine.tokenizer,
                        environment=env,
                    )
            else:
                from kanha.lam.agent import TaskAgent
                agent = TaskAgent(
                    model=engine.model,
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