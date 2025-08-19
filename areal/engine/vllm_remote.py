"""Clean Remote vLLM engine implementation (async minimal)."""

import asyncio
import os
import random
import shutil
import time
from concurrent.futures import Future, ProcessPoolExecutor
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
        self.config = config
        raw_addrs = os.getenv("AREAL_LLM_SERVER_ADDRS", "").strip()
        if not raw_addrs:
            raise RuntimeError("AREAL_LLM_SERVER_ADDRS is not set for vLLM remote.")
        self.addresses = [a.strip() for a in raw_addrs.split(",") if a.strip()]
        if not self.addresses:
            raise RuntimeError("No configured vLLM servers.")
        self.server_idx = random.randint(0, len(self.addresses) - 1)
        self.rid_to_address: Dict[str, str] = {}
        self.rid_queue: List[str] = []
        self._version = 0
        self.executor = ProcessPoolExecutor(max_workers=1)
        self.workflow_executor = WorkflowExecutor(config=config, inference_engine=self)

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
        self.executor.shutdown(wait=False, cancel_futures=True)

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

    # Internal lightweight pause flag (does NOT call remote endpoints; server side weight
    # update endpoint already aborts active requests when interrupt=True). This avoids
    # hard dependency on /pause_generation existing on server.
    def _set_paused(self, flag: bool):
        setattr(self, "_locally_paused", flag)

    def is_paused(self) -> bool:
        return getattr(self, "_locally_paused", False)

    async def agenerate(self, req: ModelRequest) -> ModelResponse:  # type: ignore[override]
        gconfig = req.gconfig
        if gconfig.n_samples != 1:
            raise ValueError("RemotevLLMEngine only supports n_samples == 1.")

        # Server selection with RID stickiness
        if req.rid in self.rid_to_address:
            server_addr = self.rid_to_address[req.rid]
        else:
            server_addr = self.choose_server()
            if len(self.rid_queue) >= RID_CACHE_SIZE:
                oldest = self.rid_queue.pop(0)
                self.rid_to_address.pop(oldest, None)
            self.rid_to_address[req.rid] = server_addr
            self.rid_queue.append(req.rid)

        tokenizer = req.tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer required for vLLM remote.")

        # Stop (optional) sequences decode
        stop_sequences: List[str] | None = None
        if gconfig.stop_token_ids:
            stop_sequences = [tokenizer.decode([tid]) for tid in gconfig.stop_token_ids]
        # NOTE: prompt payload uses token ids list (backend specific) as in validated snippet
        payload = {
            "prompt": req.input_ids,  # backend expects token ids (validated)
            "top_p": gconfig.top_p,
            "top_k": gconfig.top_k,
            "max_tokens": gconfig.max_new_tokens,
            "temperature": 0.0 if gconfig.greedy else gconfig.temperature,
            "logprobs": 1,
            "stream": False,
        }
        if stop_sequences:
            payload["stop"] = stop_sequences

        start_time = time.perf_counter()
        accumulated_output_tokens: List[int] = []
        accumulated_output_logprobs: List[float] = []
        accumulated_versions: List[int] = []
        stop_reason = "length"

        while (
            stop_reason != "stop"
            and len(accumulated_output_tokens) < gconfig.max_new_tokens
        ):
            result = await arequest_with_retry(
                session=self.workflow_executor.session,
                addr=server_addr,
                endpoint="/v1/completions",
                payload=payload,
                method="POST",
                max_retries=self.config.request_retries,
                timeout=self.config.request_timeout,
            )

            # Parse response (user-provided validated core logic)
            meta_info = result["choices"][0]
            vllm_tokens = meta_info["logprobs"]["tokens"]
            output_tokens_before = meta_info['text']  # retained for potential debug
            output_tokens = tokenizer.convert_tokens_to_ids(vllm_tokens)
            output_logprobs = meta_info["logprobs"]["token_logprobs"]

            # Update accumulated outputs
            accumulated_output_tokens.extend(output_tokens)
            accumulated_output_logprobs.extend(output_logprobs)
            accumulated_versions.extend([-1] * len(output_tokens))  # FIXME: replace with real versions when available

            stop_reason = meta_info.get("finish_reason", "stop")

            # Prepare next iteration if needed
            if (
                stop_reason != "stop"
                and len(accumulated_output_tokens) < gconfig.max_new_tokens
            ):
                payload["prompt"] = req.input_ids + accumulated_output_tokens
                payload["max_tokens"] = gconfig.max_new_tokens - len(accumulated_output_tokens)
            else:
                break

        latency = time.perf_counter() - start_time
        return ModelResponse(
            input_tokens=req.input_ids,
            input_images=req.image_data,
            output_tokens=accumulated_output_tokens,
            output_logprobs=accumulated_output_logprobs,
            output_versions=accumulated_versions,
            stop_reason=stop_reason,
            latency=latency,
            ttft=latency,
            tokenizer=req.tokenizer,
            processor=req.processor,
        )

    def update_weights(self, meta: WeightUpdateMeta) -> Future:  # type: ignore[override]
        if meta.type != "disk":
            raise NotImplementedError("Remote vLLM only supports disk weight update.")
        # Set local pause flag (remote servers may not expose pause endpoints; rely on
        # interrupt=True to abort active generations safely).
        self._set_paused(True)
        fut = self.executor.submit(
            update_weights_from_disk_vllm,
            self.config.experiment_name,
            self.config.trial_name,
            meta.model_version,
            self.addresses,
            meta.path,
            self.config.request_retries,
            self.config.request_timeout,
        )

        def _done(_f: Future):
            try:
                self.set_version(meta.model_version)
            except Exception:
                pass
            shutil.rmtree(meta.path, ignore_errors=True)
            self._set_paused(False)

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


def update_weights_from_disk_vllm(experiment_name: str, trial_name: str, model_version: int, addresses: List[str], path: str | None, request_retries: int, request_timeout: float):
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
        # Support both hyphen and underscore forms for maximum compatibility.
        update_variants = ["/update-weights-from-disk", "/update_weights_from_disk"]

        async def _call(addr: str):
            last_err: Any = None
            for ep in update_variants:
                try:
                    return await arequest_with_retry(
                        addr=addr,
                        session=session,
                        endpoint=ep,
                        payload={"path": str(path), "interrupt": True},
                        method="POST",
                        max_retries=request_retries,
                        timeout=request_timeout,
                    )
                except Exception as e:  # noqa: BLE001
                    last_err = e
            return last_err

        results = await asyncio.gather(*[_call(addr) for addr in addresses], return_exceptions=True)
        await session.close()
        failures = [r for r in results if isinstance(r, Exception) or (isinstance(r, dict) and not r.get("ok", True))]
        if failures:
            logger.warning(f"Some vLLM servers failed weight update: {failures}")
        logger.info(f"vLLM weight loading done in {(datetime.now().timestamp() - load_ts):.2f}s")
        return True
    try:
        import uvloop  # optional
        uvloop.install()
    except Exception:
        pass
    return asyncio.run(_run())
    # End of update_weights_from_disk_vllm
