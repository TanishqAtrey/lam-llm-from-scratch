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

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

from kanha.inference.engine import InferenceEngine
from kanha.utils.logging import get_logger

log = get_logger(__name__)

# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
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

def create_app(model_path: str, index_dir: str = None) -> FastAPI:
    app = FastAPI(
        title="KANHA AI API",
        description="Knowledge-Augmented Neural Heuristic Assistant",
        version="0.1.0",
    )

    # Load engine once at startup
    engine = InferenceEngine.from_pretrained(
        model_path=model_path,
        index_dir=index_dir,
    )

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model_path}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest):
        if not req.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty.")
        try:
            response = engine.chat(req.message, stream=False)
            return ChatResponse(response=response)
        except Exception as e:
            log.error(f"Chat error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/reset")
    def reset():
        engine.reset_memory()
        return {"status": "memory cleared"}

    @app.post("/agent", response_model=AgentResponse)
    def agent_endpoint(req: AgentRequest):
        """Runs a LAM agent trajectory for the given goal."""
        if not req.goal.strip():
            raise HTTPException(status_code=400, detail="Goal cannot be empty.")
        try:
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
        except Exception as e:
            log.error(f"Agent error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app