#!/usr/bin/env python3
"""
Free LLM API Router - Route requests across multiple free LLM providers.
OpenAI-compatible /v1/chat/completions endpoint with automatic failover,
smart capability-based routing (vision, large context, tool calling),
content normalization, multi-key round-robin pooling, and resilient stream delivery.
"""

import asyncio
import time
import json
import os
import gc
import datetime
import yaml
import httpx
import httpcore
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field


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
            clean_error = error.replace("\r", " ").replace("\n", " ").strip()
            log_line += f" | Error={clean_error}"

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
                    line = line.strip()
                    if not line or not line.startswith("["):
                        continue
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


@dataclass
class KeyState:
    key: str
    rate_limited_until: float = 0
    errors: int = 0
    requests_made: int = 0


@dataclass
class ProviderState:
    name: str
    base_url: str
    api_keys: List[KeyState] = field(default_factory=list)
    models: list = field(default_factory=list)
    priority: int = 1
    supports_vision: bool = False
    max_context: int = 128000
    requests_made: int = 0
    last_used: float = 0
    rate_limited_until: float = 0
    errors: int = 0
    avg_latency: float = 0
    key_idx: int = 0

    def get_active_key(self) -> Optional[KeyState]:
        now = time.time()
        available_keys = [k for k in self.api_keys if k.rate_limited_until < now]
        if not available_keys:
            return None
        selected = available_keys[self.key_idx % len(available_keys)]
        self.key_idx += 1
        return selected

    @property
    def api_key(self) -> str:
        k = self.get_active_key()
        return k.key if k else (self.api_keys[0].key if self.api_keys else "")


class Router:
    def __init__(self, config_path="config.yaml"):
        self.providers: list[ProviderState] = []
        self.strategy = "smart-priority"
        self.failover = True
        self.max_retries = 5
        self.timeout = 45
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
            raw_keys = p.get("api_keys") or ([p["api_key"]] if "api_key" in p else [])
            valid_keys = [
                KeyState(key=str(k).strip())
                for k in raw_keys
                if k and not str(k).startswith("YOUR_")
            ]
            if not valid_keys:
                continue  # Skip unconfigured providers

            self.providers.append(ProviderState(
                name=p["name"],
                base_url=p["base_url"],
                api_keys=valid_keys,
                models=p.get("models", []),
                priority=p.get("priority", 99),
                supports_vision=bool(p.get("supports_vision", False)),
                max_context=int(p.get("max_context", 128000)),
            ))

        routing = cfg.get("routing", {})
        self.strategy = routing.get("strategy", "smart-priority")
        self.failover = routing.get("failover", True)
        self.max_retries = routing.get("max_retries", 5)
        self.timeout = routing.get("timeout", 45)

    def get_provider(
        self,
        model: str = "auto",
        exclude: set = None,
        has_images: bool = False,
        estimated_tokens: int = 0,
    ) -> Optional[ProviderState]:
        now = time.time()
        exclude = exclude or set()

        candidates = [
            p for p in self.providers
            if p.rate_limited_until < now
            and any(k.rate_limited_until < now for k in p.api_keys)
            and p.name not in exclude
        ]
        if not candidates:
            return None

        # 1. Vision constraint: If request contains images, restrict to vision-capable providers
        if has_images:
            vision_candidates = [p for p in candidates if p.supports_vision]
            if vision_candidates:
                candidates = vision_candidates
            else:
                candidates = [p for p in candidates if p.name in ["google", "openrouter"]]

        # 2. Context constraint: If prompt is large (>4000 tokens), avoid low-context providers (e.g. Groq free tier)
        if estimated_tokens > 4000:
            large_ctx_candidates = [p for p in candidates if p.max_context >= estimated_tokens]
            if large_ctx_candidates:
                candidates = large_ctx_candidates

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
        elif self.strategy in ["priority", "smart-priority"]:
            provider = min(available, key=lambda p: p.priority)
        elif self.strategy == "least-latency":
            provider = min(available, key=lambda p: p.avg_latency if p.avg_latency > 0 else float("inf"))
        else:
            provider = min(available, key=lambda p: p.priority)

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


def inspect_and_normalize_request(body: dict) -> Tuple[dict, dict]:
    """
    Normalizes request body and extracts key characteristics:
    - has_images: True if any message contains image_url
    - has_tools: True if request has tools or tool_calls
    - estimated_tokens: estimated prompt token count
    - normalized_body: cleaned body where text-only list content is flattened to string
    """
    messages = body.get("messages", [])
    has_images = False
    has_tools = bool(body.get("tools") or body.get("functions"))
    char_count = 0

    new_messages = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls")
        if tool_calls or role == "tool":
            has_tools = True

        if isinstance(content, list):
            # Check for multimodal image parts
            msg_has_image = any(
                isinstance(part, dict) and (part.get("type") == "image_url" or "image_url" in part)
                for part in content
            )
            if msg_has_image:
                has_images = True
                new_messages.append(msg)
                char_count += len(str(content))
            else:
                # Text-only list parts: flatten into a single string (fixes Groq 400 & Mistral 422)
                text_parts = [
                    part.get("text", "") for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                ]
                flattened_text = "".join(text_parts) if text_parts else str(content)
                new_msg = {**msg, "content": flattened_text}
                new_messages.append(new_msg)
                char_count += len(flattened_text)
        else:
            new_messages.append(msg)
            if isinstance(content, str):
                char_count += len(content)

    estimated_tokens = char_count // 3

    normalized_body = {**body, "messages": new_messages}
    meta = {
        "has_images": has_images,
        "has_tools": has_tools,
        "estimated_tokens": estimated_tokens,
    }
    return normalized_body, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    limits = httpx.Limits(max_keepalive_connections=20, max_connections=50, keepalive_expiry=60.0)
    router.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(router.timeout, connect=10.0, read=120.0),
        limits=limits,
    )
    usage_logger.cleanup_old_logs()
    gc.collect(1)
    yield
    if router.http_client:
        await router.http_client.aclose()


app = FastAPI(title="Free LLM API Router", version="1.2.0", lifespan=lifespan)


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    if request.url.path in ["/health", "/docs", "/openapi.json"]:
        return await call_next(request)

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
    raw_body = await request.json()
    model = raw_body.get("model", "auto")
    stream = raw_body.get("stream", False)
    client_ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or (request.client.host if request.client else "unknown")
    )
    start_total = time.time()

    normalized_body, meta = inspect_and_normalize_request(raw_body)
    has_images = meta["has_images"]
    estimated_tokens = meta["estimated_tokens"]

    print(
        f"📥 Request: IP={client_ip}, model='{model}', stream={stream}, "
        f"tokens~{estimated_tokens}, vision={has_images}, tools={meta['has_tools']}"
    )

    tried_providers = set()

    for attempt in range(router.max_retries):
        provider = router.get_provider(
            model=model,
            exclude=tried_providers,
            has_images=has_images,
            estimated_tokens=estimated_tokens,
        )
        if not provider:
            print("⚠️ No suitable provider available for this request.")
            usage_logger.log(
                client_ip=client_ip,
                requested_model=model,
                provider="none",
                actual_model="none",
                stream=stream,
                latency_ms=int((time.time() - start_total) * 1000),
                status_code=503,
                error="No suitable provider available (constraints not met)",
            )
            return JSONResponse(
                status_code=503,
                content={"error": {"message": "No suitable provider available or retries exhausted", "type": "server_error"}},
            )

        active_key_state = provider.get_active_key()
        if not active_key_state:
            provider.rate_limited_until = time.time() + 30
            tried_providers.add(provider.name)
            continue

        tried_providers.add(provider.name)

        is_auto = (
            model == "auto"
            or model.endswith(":auto")
            or model == "default"
            or "auto" in model.lower()
        )

        if is_auto:
            if has_images and provider.name == "openrouter":
                actual_model = "inclusionai/ling-3.0-flash-vl:free"
            elif has_images and provider.name == "google":
                actual_model = "gemini-flash-latest"
            else:
                actual_model = provider.models[0] if provider.models else "default"
        elif model.startswith(f"{provider.name}/"):
            actual_model = model[len(provider.name) + 1:]
        elif model in provider.models:
            actual_model = model
        elif any(model.endswith(f"/{m}") for m in provider.models):
            actual_model = next(m for m in provider.models if model.endswith(f"/{m}"))
        else:
            actual_model = provider.models[0] if provider.models else "default"

        key_mask = active_key_state.key[:10] + "..." + active_key_state.key[-4:]
        print(f"🔄 [Attempt {attempt+1}/{router.max_retries}] Routing to '{provider.name}' ({key_mask}) with actual_model='{actual_model}'")

        request_body = {**normalized_body, "model": actual_model}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {active_key_state.key}",
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
                    active_key_state.rate_limited_until = time.time() + 60
                    print(f"⚠️ Key {key_mask} on '{provider.name}' rate limited (429).")
                    if not any(k.rate_limited_until < time.time() for k in provider.api_keys):
                        router.mark_rate_limited(provider)
                    else:
                        tried_providers.discard(provider.name)  # Retry same provider with another key

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
                    active_key_state.errors += 1
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

                chunk_iter = resp.aiter_bytes()
                first_chunk = None
                try:
                    first_chunk = await chunk_iter.__anext__()
                except StopAsyncIteration:
                    first_chunk = b""
                except Exception as e:
                    await resp.aclose()
                    router.record_error(provider)
                    active_key_state.errors += 1
                    print(f"❌ Provider '{provider.name}' failed to yield initial chunk: {e}")
                    if router.failover:
                        continue
                    return JSONResponse(status_code=502, content={"error": {"message": f"Stream error: {e}"}})

                if b'"error"' in first_chunk and (b'"message"' in first_chunk or b'"code"' in first_chunk):
                    err_msg = first_chunk.decode("utf-8", errors="ignore")[:150]
                    await resp.aclose()
                    print(f"❌ Provider '{provider.name}' returned error chunk: {err_msg}")
                    router.record_error(provider)
                    active_key_state.errors += 1
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=True,
                        latency_ms=int((time.time() - start) * 1000),
                        status_code=502,
                        error=f"Stream error chunk: {err_msg}",
                    )
                    if router.failover:
                        continue
                    return JSONResponse(status_code=502, content={"error": {"message": err_msg}})

                latency = time.time() - start
                router.record_success(provider, latency)
                active_key_state.requests_made += 1

                async def stream_gen():
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    if first_chunk:
                        yield first_chunk
                    try:
                        async for chunk in chunk_iter:
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
                    except (httpx.RemoteProtocolError, httpx.ReadTimeout, httpcore.RemoteProtocolError, asyncio.TimeoutError) as e:
                        print(f"⚠️ Upstream stream severed by '{provider.name}': {e}. Gracefully closing SSE.")
                        yield b"\ndata: [DONE]\n\n"
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
                    active_key_state.rate_limited_until = time.time() + 60
                    print(f"⚠️ Key {key_mask} on '{provider.name}' rate limited (429).")
                    if not any(k.rate_limited_until < time.time() for k in provider.api_keys):
                        router.mark_rate_limited(provider)
                    else:
                        tried_providers.discard(provider.name)

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
                    active_key_state.errors += 1
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

                result = resp.json()

                if isinstance(result, dict) and "error" in result and result.get("error"):
                    err_obj = result["error"]
                    err_msg = err_obj.get("message", str(err_obj)) if isinstance(err_obj, dict) else str(err_obj)
                    print(f"❌ Provider '{provider.name}' returned 200 with error body: {err_msg}")
                    router.record_error(provider)
                    active_key_state.errors += 1
                    usage_logger.log(
                        client_ip=client_ip,
                        requested_model=model,
                        provider=provider.name,
                        actual_model=actual_model,
                        stream=False,
                        latency_ms=int(latency * 1000),
                        status_code=502,
                        error=f"Upstream error in 200: {err_msg}",
                    )
                    if router.failover:
                        continue
                    return JSONResponse(status_code=502, content={"error": {"message": err_msg}})

                router.record_success(provider, latency)
                active_key_state.requests_made += 1
                result["_router"] = {
                    "provider": provider.name,
                    "key": key_mask,
                    "latency_ms": int(latency * 1000)
                }

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
            active_key_state.errors += 1
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
    now = time.time()
    return {
        "providers": [
            {
                "name": p.name,
                "available": p.rate_limited_until < now and any(k.rate_limited_until < now for k in p.api_keys),
                "total_keys": len(p.api_keys),
                "active_keys": sum(1 for k in p.api_keys if k.rate_limited_until < now),
                "priority": p.priority,
                "supports_vision": p.supports_vision,
                "max_context": p.max_context,
                "requests": p.requests_made,
                "avg_latency_ms": int(p.avg_latency * 1000),
                "errors": p.errors,
                "rate_limited_for": max(0, int(p.rate_limited_until - now)),
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
