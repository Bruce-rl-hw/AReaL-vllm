import asyncio
import os
import threading
import random
import shutil
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import aiohttp
import requests
from tensordict import TensorDict
from torchdata.stateful_dataloader import StatefulDataLoader

from areal.api.cli_args import InferenceEngineConfig
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import FinetuneSpec, ModelRequest, ModelResponse, WeightUpdateMeta
from areal.api.workflow_api import RolloutWorkflow, WorkflowExecutor
from areal.utils.http import arequest_with_retry, get_default_connector
from realhf.base import logging, name_resolve, names

logger = logging.getLogger(__name__)

RID_CACHE_SIZE = 128


class RemotevLLMEngine(InferenceEngine):
    def __init__(self, config: InferenceEngineConfig):
        # Basic config and addresses
        self.config = config
        raw_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "").strip()
        if not raw_addrs:
            raise RuntimeError("AREAL_LLM_SERVER_ADDRS is not set for vLLM remote.")
        self.addresses = [a.strip() for a in raw_addrs.split(",") if a.strip()]
        if not self.addresses:
            raise RuntimeError("No configured vLLM servers.")

        # Routing / version
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.rid_to_address = {}
        self.rid_queue = []
        self._version = 0

        # Executor for background update task
        self.thread_executor = ThreadPoolExecutor(max_workers=1)

        # Workflow executor
        self.workflow_executor = WorkflowExecutor(config=config, inference_engine=self)

        # Update guards
        self._update_lock = threading.Lock()
        self._update_future = None
        self._update_version = None
        self._updating_event = threading.Event()
        # Active request tracking (drain before weight swap)
        self._active_reqs = 0
        self._active_reqs_lock = threading.Lock()

    # One-shot updates; no per-address pause/rolling logic.

    def _wait_for_server(self, address: str):
        base = f"http://{address}"
        deadline = time.time() + self.config.setup_timeout
        while time.time() < deadline:
            if self.check_health(base):
                return
            time.sleep(1)
        raise RuntimeError(f"vLLM server {address} not healthy in time.")

    def check_health(self, base_url: str) -> bool:
        for ep in ["/health", "/v1/models"]:
            try:
                resp = requests.get(base_url + ep, timeout=5)
                if resp.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
        return False

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec | None = None):  # type: ignore[override]
        logger.info("Waiting for vLLM servers ready ...")
        for a in self.addresses:
            self._wait_for_server(a)
        logger.info("vLLM servers are all ready!")
        self.workflow_executor.initialize()

    def destroy(self):  # type: ignore[override]
        self.workflow_executor.destroy()
        self.thread_executor.shutdown(wait=False, cancel_futures=True)

    def set_version(self, version: int):  # type: ignore[override]
        self._version = version

    def get_version(self) -> int:  # type: ignore[override]
        return self._version

    def choose_server(self) -> str:
        if self.config.schedule_policy == "round_robin":
            addr = self.addresses[self.server_idx]
            self.server_idx = (self.server_idx + 1) % len(self.addresses)
            return addr
        raise NotImplementedError(f"Unsupported schedule policy: {self.config.schedule_policy}")

    async def agenerate(self, req: ModelRequest, tokenizer=None) -> ModelResponse:  # type: ignore[override]
        # Wait if update in progress
        while self._updating_event.is_set():
            await asyncio.sleep(0.05)
        # Mark active
        with self._active_reqs_lock:
            self._active_reqs += 1
        try:
            gconfig = req.gconfig
            if gconfig.n_samples != 1:
                raise ValueError("RemotevLLMEngine only supports n_samples == 1.")

            if req.rid in self.rid_to_address:
                server_addr = self.rid_to_address[req.rid]
            else:
                server_addr = self.choose_server()

            if req.rid not in self.rid_to_address or self.rid_to_address[req.rid] != server_addr:
                if len(self.rid_queue) >= RID_CACHE_SIZE:
                    oldest = self.rid_queue.pop(0)
                    self.rid_to_address.pop(oldest, None)
                self.rid_to_address[req.rid] = server_addr
                self.rid_queue.append(req.rid)

            tokenizer = tokenizer or req.tokenizer
            if tokenizer is None:
                raise RuntimeError("Tokenizer required for vLLM remote.")

            stop_sequences: List[str] | None = None
            if gconfig.stop_token_ids:
                stop_sequences = [tokenizer.decode([tid]) for tid in gconfig.stop_token_ids]

            payload = {
                "prompt": req.input_ids,
                "top_p": gconfig.top_p,
                "top_k": gconfig.top_k,
                "max_tokens": gconfig.max_new_tokens,
                "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
                "logprobs": 1,
                "stream": False,
                # Ask server to return token ids directly if supported
                "return_tokens_as_token_ids": True,
            }
            if stop_sequences:
                payload["stop"] = stop_sequences

            # Log the prompt once (full decode)
            try:
                _prompt_text = tokenizer.decode(req.input_ids, skip_special_tokens=False)
            except Exception as e:
                _prompt_text = f"<decode_error: {e}>"
            logger.warning(
                "PROMPT (rid=%s len=%d)\n%s",
                getattr(req, "rid", None),
                len(req.input_ids) if isinstance(req.input_ids, list) else -1,
                _prompt_text,
            )

            start_time = time.perf_counter()
            accumulated_output_tokens, accumulated_output_logprobs, accumulated_versions = [], [], []
            stop_reason = "length"

            # --------------------------------- simplified loop ---------------------------------
            while payload["max_tokens"] > 0:
                result = await arequest_with_retry(
                    session=self.workflow_executor.session,
                    addr=server_addr,
                    endpoint="/v1/completions",
                    payload=payload,
                    method="POST",
                    max_retries=self.config.request_retries,
                    timeout=self.config.request_timeout,
                )

                meta_info = result["choices"][0]

                # ---- finish / abort handling ----
                finish_reason = meta_info["finish_reason"]
                stop_reason = finish_reason  # keep the same naming as the reference snippet
                if (
                    stop_reason == "abort"
                    and isinstance(finish_reason, dict)
                    and finish_reason.get("message") == "Abort before prefill"
                ):
                    continue  # retry the loop immediately

                # ---- parse tokens ----
                output_tokens_raw = meta_info["logprobs"]["tokens"]
                output_tokens = [int(t.split(":")[1]) for t in output_tokens_raw]
                output_logprobs = meta_info["logprobs"]["token_logprobs"]

                # ---- accumulate ----
                accumulated_output_tokens.extend(output_tokens)
                accumulated_output_logprobs.extend(output_logprobs)
                accumulated_versions.extend([-1] * len(output_tokens))

                 # ——放在你每轮 result/choice 处理后（accumulated_output_tokens 已更新）——
                srv_text = meta_info.get("text", "") or ""
                try:
                    client_decoded = tokenizer.decode(
                        accumulated_output_tokens, skip_special_tokens=False
                    )
                except Exception as e:
                    client_decoded = f"<decode_error: {e}"

                # 只截前 300 个可视字符，换行显示成 ⏎，避免刷屏
                def _head(s: str, n: int = 300) -> str:
                    try:
                        s = s.replace("\n", " ⏎ ")
                    except Exception:
                        pass
                    return (s[:n] + (" …" if len(s) > n else "")) if isinstance(s, str) else str(s)

                logger.warning(
                    "=== TEXT vs DECODE (rid=%s addr=%s step_tokens=%d stop=%s) ===\n"
                    "[server text] %s\n"
                    "[client decode] %s",
                    getattr(req, "rid", None),
                    server_addr,
                    len(accumulated_output_tokens),
                    stop_reason,
                    _head(srv_text),
                    _head(client_decoded),
                )

                if (
                    stop_reason not in ["stop", "abort"]
                    and len(accumulated_output_tokens) < gconfig.max_new_tokens
                ):
                    payload["prompt"] = req.input_ids + accumulated_output_tokens
                    payload["max_tokens"] = gconfig.max_new_tokens - len(accumulated_output_tokens)
                else:
                    break
            # --------------------------------- simplified loop end -----------------------------

            latency = time.perf_counter() - start_time
            return ModelResponse(
                input_tokens=req.input_ids,
                input_images=req.image_data,
                output_tokens=accumulated_output_tokens,
                output_logprobs=accumulated_output_logprobs,
                output_versions=accumulated_versions,
                stop_reason=stop_reason,
                latency=latency,
                ttft=latency,  # non-streaming
                tokenizer=req.tokenizer,
                processor=req.processor,
            )
        finally:
            with self._active_reqs_lock:
                self._active_reqs -= 1

    def update_weights(self, meta: WeightUpdateMeta) -> Future:  # type: ignore[override]
        """
        Update weights on remote vLLM servers.
        
        IMPORTANT: In distributed training (d2p1t1, etc.), this method should ONLY be called by rank 0!
        All other ranks should wait for rank 0 to complete the remote update before proceeding.
        
        Correct pattern:
            if dist.get_rank() == 0:
                future = rollout.update_weights(meta)
            actor.upload_weights(meta)  # all ranks
            if dist.get_rank() == 0:
                future.result()
        """
        if meta.type != "disk":
            raise NotImplementedError("Remote vLLM only supports disk weight update.")

        with self._update_lock:
            # Version validation
            if not isinstance(meta.model_version, int):
                raise ValueError(f"invalid model_version type={type(meta.model_version)}")
            if meta.model_version < 0:
                raise ValueError(f"negative model_version={meta.model_version} not allowed")

            # Fix version issue: if meta.model_version is 0 (default), use engine's current version
            # This matches SGLang's behavior: use self.get_version() directly
            actual_version = meta.model_version if meta.model_version > 0 else self.get_version()
            
            # Simple deduplication - wait for existing updates to complete
            if self._update_future and not self._update_future.done():
                logger.info(f"[vllm_remote][update] waiting for existing update to complete before starting v{actual_version}")
                try:
                    self._update_future.result(timeout=60.0)
                except Exception as e:
                    logger.warning(f"[vllm_remote][update] previous update failed: {e}")
                finally:
                    self._update_future = None
                    self._update_version = None

            logger.info(f"[vllm_remote][update] starting weight update version={actual_version}")

            def _update_wrapper():
                try:
                    # Pause requests
                    self._updating_event.set()
                    self.pause()
                    
                    # Wait for active requests to drain (simple timeout)
                    deadline = time.time() + 10.0
                    while time.time() < deadline:
                        with self._active_reqs_lock:
                            if self._active_reqs == 0:
                                break
                        time.sleep(0.1)
                    
                    # Clear routing cache
                    self.rid_to_address.clear()
                    self.rid_queue.clear()
                    
                    # Update weights on all servers directly - no filtering needed
                    update_weights_from_disk_vllm(
                        self.config.experiment_name,
                        self.config.trial_name,
                        actual_version,  # Use actual version instead of meta.model_version
                        self.addresses,
                        meta.path,
                        self.config.request_retries,
                        self.config.request_timeout,
                    )
                    
                    return True
                    
                finally:
                    # Always resume operations
                    self._updating_event.clear()
                    try:
                        self.resume()
                    except Exception as e:
                        logger.warning(f"[vllm_remote][update] resume failed: {e}")

            fut = self.thread_executor.submit(_update_wrapper)

            def _done(_f: Future):
                try:
                    if _f.result():
                        logger.info(f"[vllm_remote][update] update v{actual_version} completed successfully")
                        self.set_version(actual_version)
                    else:
                        logger.error(f"[vllm_remote][update] update v{actual_version} failed")
                except Exception as e:
                    logger.error(f"[vllm_remote][update] update v{actual_version} failed: {e}")
                
                with self._update_lock:
                    if self._update_future is _f:
                        self._update_future = None
                        self._update_version = None

            self._update_future = fut
            self._update_version = actual_version
            fut.add_done_callback(_done)
            return fut

    def submit(self, data: Dict[str, Any], workflow: Optional[RolloutWorkflow] = None, workflow_builder: Optional[Callable] = None) -> None:  # type: ignore[override]
        return self.workflow_executor.submit(data, workflow, workflow_builder)

    def wait(self, count: int, timeout: float | None = None, should_accept: Callable | None = None) -> TensorDict:  # type: ignore[override]
        return self.workflow_executor.wait(count, timeout=timeout, should_accept=should_accept)

    def rollout_batch(self, data: List[Dict[str, Any]], workflow: Optional[RolloutWorkflow] = None, workflow_builder: Optional[Callable] = None) -> TensorDict:  # type: ignore[override]
        return self.workflow_executor.rollout_batch(data, workflow, workflow_builder)

    def prepare_batch(self, dataloader: StatefulDataLoader, workflow: Optional[RolloutWorkflow] = None, workflow_builder: Optional[Callable] = None, should_accept: Callable | None = None):  # type: ignore[override]
        return self.workflow_executor.prepare_batch(dataloader, workflow, workflow_builder, should_accept)

    def pause(self):  # type: ignore[override]
        return self.workflow_executor.pause()

    def resume(self):  # type: ignore[override]
        return self.workflow_executor.resume()


def update_weights_from_disk_vllm(
    experiment_name: str,
    trial_name: str,
    model_version: int,
    addresses: List[str],
    path: str | None,
    request_retries: int,
    request_timeout: float,
):
    if path is None:
        raise RuntimeError("WeightUpdateMeta.path is None for disk update.")

    async def _run():
        update_name = names.update_weights_from_disk(experiment_name, trial_name, model_version)
        try:
            save_ts = float(name_resolve.wait(update_name, timeout=120))
        except Exception:
            save_ts = time.time()
        load_ts = datetime.now().timestamp()
        logger.info(f"Begin vLLM weight update from {path}, responded in {load_ts - save_ts:.2f}s")

        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request_timeout, sock_connect=request_timeout, connect=request_timeout),
            read_bufsize=1024 * 1024 * 8,
            connector=get_default_connector(),
        )

        results = []
        for addr in addresses:
            # Determine the appropriate payload based on available files
            payload = {"path": str(path), "interrupt": True}
            
            # Check if we have single model.safetensors or sharded files
            model_safetensors_path = os.path.join(path, "model.safetensors")
            if os.path.exists(model_safetensors_path):
                # Use single file loading with custom pattern
                payload["pattern"] = "model.safetensors"
                logger.info(f"Loading from single model.safetensors file at {path}")
            else:
                # Default to sharded pattern (will use vLLM's DEFAULT_PATTERN)
                logger.info(f"Loading from sharded files at {path}")
            
            r = await arequest_with_retry(
                addr=addr,
                session=session,
                endpoint="/update-weights-from-disk",
                payload=payload,
                method="POST",
                max_retries=request_retries,
                timeout=request_timeout,
            )
            results.append(r)

        await session.close()

        failures = [r for r in results if isinstance(r, dict) and not r.get("ok", True)]
        if failures:
            raise RuntimeError(f"weight update failures: {failures}")

        logger.info(f"vLLM weight loading done in {(datetime.now().timestamp() - load_ts):.2f}s")
        return True

    return asyncio.run(_run())

