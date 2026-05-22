#!/usr/bin/env python3
"""Adaptive GPU-utilization filler.

Runs lightweight GEMM work on visible CUDA devices when their recent GPU
utilization is below a configured threshold. Intended as a sidecar process
inside an existing Slurm allocation on the same node as a training job.
"""

from __future__ import annotations

import argparse
import collections
import logging
import os
import signal
import sys
import threading
import time

import pynvml
import torch


_STOP = False


def _handle_stop(signum, frame):
    del signum, frame
    global _STOP
    _STOP = True


def _parse_visible_device_tokens() -> list[str]:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cuda_visible_devices.strip():
        return [str(idx) for idx in range(torch.cuda.device_count())]
    return [token.strip() for token in cuda_visible_devices.split(",") if token.strip()]


def _resolve_nvml_handle(token: str):
    if token.isdigit():
        return pynvml.nvmlDeviceGetHandleByIndex(int(token)), token
    return pynvml.nvmlDeviceGetHandleByUUID(token.encode("utf-8")), token


def _select_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability(0)
        if major >= 8:
            return torch.bfloat16
    return torch.float16


def _build_device_states(matrix_size: int):
    states = []
    dtype = _select_dtype()
    for local_idx, token in enumerate(_parse_visible_device_tokens()):
        handle, label = _resolve_nvml_handle(token)
        device = torch.device(f"cuda:{local_idx}")
        with torch.cuda.device(device):
            stream = torch.cuda.Stream(device=device)
            a = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
            b = torch.randn((matrix_size, matrix_size), device=device, dtype=dtype)
            c = torch.empty((matrix_size, matrix_size), device=device, dtype=dtype)
        states.append(
            {
                "local_idx": local_idx,
                "label": label,
                "device": device,
                "handle": handle,
                "stream": stream,
                "a": a,
                "b": b,
                "c": c,
                "active": False,
                "history": None,
                "last_gpu_util": 0.0,
                "last_mem_util": 0.0,
                "rolling_gpu_util": 0.0,
            }
        )
    return states


def _sample_utils(states):
    for state in states:
        rates = pynvml.nvmlDeviceGetUtilizationRates(state["handle"])
        gpu_util = float(rates.gpu)
        mem_util = float(rates.memory)
        history = state["history"]
        history.append(gpu_util)
        state["last_gpu_util"] = gpu_util
        state["last_mem_util"] = mem_util
        state["rolling_gpu_util"] = sum(history) / len(history)


def _fill_worker(
    state,
    run_event: threading.Event,
    stop_event: threading.Event,
    fill_iters: int,
    idle_sleep_sec: float,
):
    device = state["device"]
    stream = state["stream"]
    while not stop_event.is_set():
        if not run_event.is_set():
            time.sleep(idle_sleep_sec)
            continue
        with torch.cuda.device(device), torch.cuda.stream(stream):
            for _ in range(fill_iters):
                torch.mm(state["a"], state["b"], out=state["c"])
                state["a"], state["b"], state["c"] = state["b"], state["c"], state["a"]


def _format_summary(states) -> str:
    parts = []
    for state in states:
        status = "on" if state["active"] else "off"
        parts.append(
            f"gpu{state['local_idx']}[{state['label']}]="
            f"cur:{state['last_gpu_util']:.1f}% "
            f"avg:{state['rolling_gpu_util']:.1f}% "
            f"mem:{state['last_mem_util']:.1f}% "
            f"fill:{status}"
        )
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive GPU-utilization filler")
    parser.add_argument("--threshold", type=float, default=70.0, help="Enable filler when rolling GPU util is below this percent")
    parser.add_argument("--stop-threshold", type=float, default=None, help="Disable filler after rolling util recovers above this percent")
    parser.add_argument("--poll-sec", type=float, default=1.0, help="Seconds between utilization samples")
    parser.add_argument("--window", type=int, default=12, help="Rolling sample window used to decide when to fill")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matrix dimension for GEMM work")
    parser.add_argument("--fill-iters", type=int, default=16, help="Number of GEMMs each active worker launches per loop")
    parser.add_argument("--idle-sleep-sec", type=float, default=0.02, help="Worker sleep when filler is inactive on a GPU")
    parser.add_argument("--log-sec", type=float, default=30.0, help="How often to print a status line")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run GPU filler")

    stop_threshold = args.stop_threshold
    if stop_threshold is None:
        stop_threshold = min(100.0, args.threshold + 8.0)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    torch.set_grad_enabled(False)
    torch.set_num_threads(1)
    try:
        os.nice(10)
    except OSError:
        pass

    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    pynvml.nvmlInit()
    try:
        states = _build_device_states(args.matrix_size)
        for state in states:
            state["history"] = collections.deque(maxlen=max(1, args.window))

        stop_event = threading.Event()
        workers = []
        for state in states:
            run_event = threading.Event()
            state["run_event"] = run_event
            worker = threading.Thread(
                target=_fill_worker,
                args=(state, run_event, stop_event, args.fill_iters, args.idle_sleep_sec),
                daemon=True,
            )
            worker.start()
            workers.append(worker)

        logging.info(
            "Started GPU filler on %d visible GPUs: start_threshold=%.1f stop_threshold=%.1f poll=%.2fs window=%d matrix=%d fill_iters=%d",
            len(states),
            args.threshold,
            stop_threshold,
            args.poll_sec,
            args.window,
            args.matrix_size,
            args.fill_iters,
        )

        last_log = 0.0
        while not _STOP:
            _sample_utils(states)

            for state in states:
                rolling = state["rolling_gpu_util"]
                current = state["last_gpu_util"]
                active = state["active"]

                if not active and rolling < args.threshold:
                    state["active"] = True
                    state["run_event"].set()
                elif active and rolling >= stop_threshold and current >= args.threshold:
                    state["active"] = False
                    state["run_event"].clear()

            now = time.monotonic()
            if now - last_log >= args.log_sec:
                logging.info("Current utilization: %s", _format_summary(states))
                last_log = now

            time.sleep(args.poll_sec)
    finally:
        stop_event = locals().get("stop_event")
        if stop_event is not None:
            stop_event.set()
        for state in locals().get("states", []):
            run_event = state.get("run_event")
            if run_event is not None:
                run_event.set()
        for worker in locals().get("workers", []):
            worker.join(timeout=1.0)
        pynvml.nvmlShutdown()
        logging.info("GPU filler exiting")

    return 0


if __name__ == "__main__":
    sys.exit(main())
