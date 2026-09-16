#!/usr/bin/env python3
"""
Free LLM API Router - Route requests across multiple free LLM providers.
OpenAI-compatible /v1/chat/completions endpoint with automatic failover.
"""

import asyncio
import time
import json
import os
import gc
import datetime
import yaml
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass


class UsageLogger:
    def __init__(self, logs_dir="logs", retention_days=3):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.cleanup_old_logs()

    def get_today_log_path(self) -> Path:
        date_str = datetime.datetime.now().strftime("%Y-%m-%d")
        return self.logs_dir / f"{date_str}.log"

    def cleanup_old_logs(self):
        """Keep only the newest `retention_days` daily log files. Files older than 3 days are deleted."""
        try:
            log_files = sorted(self.logs_dir.glob("*.log"), reverse=True)
            if len(log_files) > self.retention_days:
                for old_file in log_files[self.retention_days:]:
                    old_file.unlink(missing_ok=True)
                    print(f"🗑️ Deleted expired log file (> {self.retention_days} days): {old_file.name}")
        except Exception as e:
            print(f"⚠️ Error cleaning old logs: {e}")

    def log(
        self,
        client_ip: str,
        requested_model: str,
        provider: str,
        actual_model: str,
        stream: bool,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: int = 0,
        status_code: int = 200,
        error: Optional[str] = None,
    ):
        self.cleanup_old_logs()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"[{now_str}] IP={client_ip} | ReqModel={requested_model} | "
            f"Provider={provider} | Model={actual_model} | Stream={stream} | "
            f"Tokens: prompt={prompt_tokens}, completion={completion_tokens}, total={total_tokens} | "
            f"Latency={latency_ms}ms | Status={status_code}"
        )
        if error:
            log_line += f" | Error={error}"

        log_path = self.get_today_log_path()
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except Exception as e:
            print(f"⚠️ Failed to write usage log: {e}")

    def get_summary(self) -> Dict[str, Any]:
        self.cleanup_old_logs()
        today_file = self.get_today_log_path()
        files = [
            {"name": f.name, "size_bytes": f.stat().st_size}
            for f in sorted(self.logs_dir.glob("*.log"), reverse=True)
        ]
        model_stats: Dict[str, Dict[str, int]] = {}
        total_requests = 0

        if today_file.exists():
            with open(today_file, "r", encoding="utf-8") as f:
                for line in f:
                    total_requests += 1
                    try:
                        parts = [p.strip() for p in line.split("|")]
                        model_part = next((p for p in parts if p.startswith("Model=")), "")
                        model_name = model_part.split("=", 1)[1] if model_part else "other"

                        tok_part = next((p for p in parts if p.startswith("Tokens:")), "")
                        total_tok = 0
                        if "total=" in tok_part:
                            total_tok = int(tok_part.split("total=")[1].split()[0].replace(",", ""))

                        if model_name not in model_stats:
                            model_stats[model_name] = {"requests": 0, "total_tokens": 0}
                        model_stats[model_name]["requests"] += 1
                        model_stats[model_name]["total_tokens"] += total_tok
                    except Exception:
                        continue

        return {
            "today": datetime.datetime.now().strftime("%Y-%m-%d"),
            "total_requests_today": total_requests,
            "models_usage_today": model_stats,
            "retained_log_files": files,
        }


usage_logger = UsageLogger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global persistent connection pool with Keep-Alive
    limits = httpx.Limits(max_keepalive_connections=15, max_connections=40, keepalive_expiry=60.0)
    router.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(router.timeout, connect=10.0, read=120.0),
        limits=limits,
    )
    usage_logger.cleanup_old_logs()
    gc.collect(1)
    yield
    # Clean shutdown of connection pool
    if router.http_client:
        await router.http_client.aclose()


app = FastAPI(title="Free LLM API Router", version="1.0.0", lifespan=lifespan)


@dataclass
class ProviderState:
    name: str
    base_url: str
    api_key: str
    models: list
    priority: int = 1
    requests_made: int = 0
    last_used: float = 0
    rate_limited_until: float = 0
    errors: int = 0
    avg_latency: float = 0


class Router:
    def __init__(self, config_path="config.yaml"):
        self.providers: list[ProviderState] = []
        self.strategy = "round-robin"
        self.failover = True
        self.max_retries = 3
        self.timeout = 30
        self.api_key = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.current_idx = 0
        self.load_config(config_path)

    def load_config(self, path):
        config_file = Path(path)
        if not config_file.exists():
            config_file = Path("config.example.yaml")
        if config_file.exists():
            with open(config_file) as f:
                cfg = yaml.safe_load(f) or {}
        else:
            cfg = {"providers": [], "routing": {}}

        self.api_key = cfg.get("api_key") or os.environ.get("ROUTER_API_KEY")

        for p in cfg.get("providers", []):
            key = p.get("api_key", "")
            if key.startswith("YOUR_"):
                continue  # Skip unconfigured providers
            self.providers.append(ProviderState(
                name=p["name"],
                base_url=p["base_url"],
                api_key=key,
                models=p.get("models", []),
                priority=p.get("priority", 99),
            ))

        routing = cfg.get("routing", {})
        self.strategy = routing.get("strategy", "round-robin")
        self.failover = routing.get("failover", True)
        self.max_retries = routing.get("max_retries", 3)
        self.timeout = routing.get("timeout", 30)

    def get_provider(self, model: str = "auto", exclude: set = None) -> Optional[ProviderState]:
        now = time.time()
        exclude = exclude or set()
        candidates = [p for p in self.providers if p.rate_limited_until < now and p.name not in exclude]
        if not candidates:
            return None

        is_auto = (
            model == "auto"
            or model.endswith(":auto")
            or model == "default"
            or "auto" in model.lower()
        )

        available = [
            p for p in candidates
            if is_auto
            or model in p.models
            or model.startswith(f"{p.name}/")
            or any(model.startswith(m.split("/")[0]) for m in p.models)
        ]
        if not available:
            available = candidates

        if self.strategy == "round-robin":
            provider = available[self.current_idx % len(available)]
            self.current_idx += 1
        elif self.strategy == "priority":
            provider = min(available, key=lambda p: p.priority)
        elif self.strategy == "least-latency":
            provider = min(available, key=lambda p: p.avg_latency if p.avg_latency > 0 else float("inf"))
        else:
            provider = available[self.current_idx % len(available)]
            self.current_idx += 1

        return provider

    def mark_rate_limited(self, provider: ProviderState, retry_after: int = 60):
        provider.rate_limited_until = time.time() + retry_after

    def record_success(self, provider: ProviderState, latency: float):
        provider.requests_made += 1
        provider.last_used = time.time()
        provider.errors = 0
        provider.avg_latency = (
            0.7 * provider.avg_latency + 0.3 * latency
            if provider.avg_latency > 0
            else latency
        )

    def record_error(self, provider: ProviderState):
        provider.errors += 1


router = Router()


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    # Public endpoints
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)

    # If router has an api_key configured, enforce Bearer token verification
    if router.api_key:
        auth_header = request.headers.get("Authorization", "")
        token = ""
        if auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        elif auth_header:
            token = auth_header.strip()

        if not token or token != router.api_key:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API key. Provide a valid Bearer token in the Authorization header.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
            )

    return await call_next(request)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model = body.get("model", "auto")
    stream = body.get("stream", False)
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    start_total = time.time()
    print(f"📥 Received request: IP={client_ip}, raw_model='{model}', stream={stream}")

    tried_providers = set()
    for attempt in range(router.max_retries):
        provider = router.get_provider(model, exclude=tried_providers)
        if not provider:
            print("⚠️ No more available providers to try.")
            usage_logger.log(
                client_ip=client_ip,
                requested_model=model,
                provider="none",
                actual_model="none",
                stream=stream,
                latency_ms=int((time.time() - start_total) * 1000),
                status_code=503,
                error="All providers unavailable or retries exhausted",
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "All providers unavailable or retries exhausted", "type": "server_error"}},
            )
        tried_providers.add(provider.name)

        # Map requested model to actual provider model name
        is_auto = (
            model == "auto"
            or model.endswith(":auto")
            or model == "default"
            or "auto" in model.lower()
        )

        if is_auto:
            actual_model = provider.models[0] if provider.models else "default"
        elif model.startswith(f"{provider.name}/"):
            actual_model = model[len(provider.name) + 1:]
        elif model in provider.models:
            actual_model = model
        elif any(model.endswith(f"/{m}") for m in provider.models):
            actual_model = next(m for m in provider.models if model.endswith(f"/{m}"))
        else:
            actual_model = provider.models[0] if provider.models else "default"

        print(f"🔄 [Attempt {attempt+1}/{router.max_retries}] Routing to '{provider.name}' with actual_model='{actual_model}'")

        request_body = {**body, "model": actual_model}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.api_key}",
        }

        start = time.time()
        client = router.http_client or httpx.AsyncClient(timeout=httpx.Timeout(router.timeout, connect=10.0, read=120.0))
        try:
            if stream:
                req = client.build_request(
                    "POST",
                    f"{provider.base_url}/chat/completions",
                    json=request_body,
                    headers=headers,
                )
                resp = await client.send(req, stream=True)

                if resp.status_code == 429:
                    await resp.aclose()
                    print(f"⚠️ Provider '{provider.name}' rate limited (429). Failing over...")
                    router.mark_rate_limited(provider)
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=True,
                        latency_ms=int((time.time() - start) * 1000),
                        status_code=429,
                        error="Rate limited",
                    )
                    if router.failover:
                        continue
                    return JSONResponse(status_code=429, content={"error": {"message": "Rate limited"}})

                if resp.status_code != 200:
                    err_bytes = await resp.aread()
                    await resp.aclose()
                    err_msg = err_bytes.decode("utf-8", errors="ignore")[:150]
                    print(f"❌ Provider '{provider.name}' error ({resp.status_code}): {err_msg}")
                    router.record_error(provider)
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=True,
                        latency_ms=int((time.time() - start) * 1000),
                        status_code=resp.status_code,
                        error=err_msg,
                    )
                    if router.failover:
                        continue
                    return JSONResponse(
                        status_code=resp.status_code,
                        content={"error": {"message": f"Provider error: {resp.status_code} - {err_msg}"}},
                    )

                latency = time.time() - start
                router.record_success(provider, latency)

                async def stream_gen():
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    try:
                        async for chunk in resp.aiter_bytes():
                            if b'"usage"' in chunk:
                                try:
                                    for line in chunk.split(b"\n"):
                                        line = line.strip()
                                        if line.startswith(b"data: ") and line != b"data: [DONE]":
                                            payload = json.loads(line[6:])
                                            u = payload.get("usage")
                                            if u:
                                                prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                                                completion_tokens = u.get("completion_tokens", completion_tokens)
                                                total_tokens = u.get("total_tokens", prompt_tokens + completion_tokens)
                                except Exception:
                                    pass
                            yield chunk
                    finally:
                        await resp.aclose()
                        latency_ms = int((time.time() - start) * 1000)
                        usage_logger.log(
                            client_ip=client_ip,
                            requested_model=model,
                            provider=provider.name,
                            actual_model=actual_model,
                            stream=True,
                            prompt_tokens=prompt_tokens,
                            completion_tokens=completion_tokens,
                            total_tokens=total_tokens,
                            latency_ms=latency_ms,
                            status_code=200,
                        )

                return StreamingResponse(stream_gen(), media_type="text/event-stream")
            else:
                resp = await client.post(
                    f"{provider.base_url}/chat/completions",
                    json=request_body,
                    headers=headers,
                )
                latency = time.time() - start

                if resp.status_code == 429:
                    router.mark_rate_limited(provider)
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=False,
                        latency_ms=int((time.time() - start) * 1000),
                        status_code=429,
                        error="Rate limited",
                    )
                    if router.failover:
                        continue
                    return JSONResponse(status_code=429, content={"error": {"message": "Rate limited"}})

                if resp.status_code != 200:
                    router.record_error(provider)
                    err_msg = resp.text[:150]
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=False,
                        latency_ms=int((time.time() - start) * 1000),
                        status_code=resp.status_code,
                        error=err_msg,
                    )
                    if router.failover:
                        continue
                    return JSONResponse(status_code=resp.status_code, content=resp.json())

                router.record_success(provider, latency)
                result = resp.json()
                result["_router"] = {"provider": provider.name, "latency_ms": int(latency * 1000)}

                u = result.get("usage", {}) or {}
                p_tok = u.get("prompt_tokens", 0)
                c_tok = u.get("completion_tokens", 0)
                t_tok = u.get("total_tokens", p_tok + c_tok)
                usage_logger.log(
                    client_ip=client_ip,
                    requested_model=model,
                    provider=provider.name,
                    actual_model=actual_model,
                    stream=False,
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    latency_ms=int(latency * 1000),
                    status_code=200,
                )
                return JSONResponse(content=result)

        except Exception as e:
            router.record_error(provider)
            if router.failover and attempt < router.max_retries - 1:
                continue
            usage_logger.log(
                client_ip=client_ip,
                requested_model=model,
                provider=provider.name,
                actual_model=actual_model,
                stream=stream,
                latency_ms=int((time.time() - start) * 1000),
                status_code=502,
                error=str(e),
            )
            return JSONResponse(status_code=502, content={"error": {"message": str(e)}})

    usage_logger.log(
        client_ip=client_ip,
        requested_model=model,
        provider="none",
        actual_model="none",
        stream=stream,
        latency_ms=int((time.time() - start_total) * 1000),
        status_code=503,
        error="All retries exhausted",
    )
    return JSONResponse(status_code=503, content={"error": {"message": "All retries exhausted"}})


@app.get("/usage")
@app.get("/v1/usage")
async def get_usage():
    return usage_logger.get_summary()


@app.get("/v1/models")
async def list_models():
    models = []
    for p in router.providers:
        for m in p.models:
            models.append({"id": f"{p.name}/{m}", "object": "model", "owned_by": p.name})
    models.append({"id": "auto", "object": "model", "owned_by": "router"})
    return {"object": "list", "data": models}


@app.get("/status")
async def status():
    return {
        "providers": [
            {
                "name": p.name,
                "available": p.rate_limited_until < time.time(),
                "requests": p.requests_made,
                "avg_latency_ms": int(p.avg_latency * 1000),
                "errors": p.errors,
                "rate_limited_for": max(0, int(p.rate_limited_until - time.time())),
            }
            for p in router.providers
        ],
        "strategy": router.strategy,
        "total_providers": len(router.providers),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "providers": len(router.providers)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    print(f"🚀 Free LLM API Router starting on port {port}")
    print(f"📊 {len(router.providers)} providers loaded")
    print(f"🔄 Strategy: {router.strategy}")
    if router.api_key:
        masked = router.api_key[:6] + "..." + router.api_key[-4:] if len(router.api_key) > 10 else "***"
        print(f"🔒 Authentication: ENABLED (Key: {masked})")
    else:
        print("⚠️ Authentication: DISABLED (Public access)")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        limit_concurrency=40,
        timeout_keep_alive=30,
        access_log=False,
    )
