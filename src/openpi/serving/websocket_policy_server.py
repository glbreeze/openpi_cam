import asyncio
import dataclasses
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _PendingInference:
    obs: dict
    response_future: asyncio.Future
    remote_address: tuple[str, int] | None


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        batch_max_size: int = 1,
        batch_timeout_ms: float = 0.0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._batch_max_size = max(1, batch_max_size)
        self._batch_timeout_s = max(0.0, batch_timeout_ms / 1000.0)
        self._request_queue: asyncio.Queue[_PendingInference] | None = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        self._request_queue = asyncio.Queue()
        batch_worker = asyncio.create_task(self._batch_worker())
        try:
            async with _server.serve(
                self._handler,
                self._host,
                self._port,
                compression=None,
                max_size=None,
                process_request=_health_check,
            ) as server:
                await server.serve_forever()
        finally:
            batch_worker.cancel()
            await asyncio.gather(batch_worker, return_exceptions=True)

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                logger.info("Received inference request from %s", websocket.remote_address)

                if self._request_queue is None:
                    raise RuntimeError("Inference queue is not initialized.")
                response_future = asyncio.get_running_loop().create_future()
                await self._request_queue.put(_PendingInference(obs, response_future, websocket.remote_address))
                action = await response_future
                infer_time = time.monotonic() - start_time
                logger.info(
                    "Completed inference request from %s in %.2f ms",
                    websocket.remote_address,
                    infer_time * 1000,
                )

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    async def _batch_worker(self) -> None:
        if self._request_queue is None:
            raise RuntimeError("Inference queue is not initialized.")

        while True:
            first_request = await self._request_queue.get()
            batch = [first_request]
            deadline = time.monotonic() + self._batch_timeout_s

            while len(batch) < self._batch_max_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    next_request = await asyncio.wait_for(self._request_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    break
                batch.append(next_request)

            batch_start = time.monotonic()
            try:
                results = await asyncio.to_thread(self._infer_batch, [request.obs for request in batch])
                if len(results) != len(batch):
                    raise RuntimeError(f"Expected {len(batch)} batch results, got {len(results)}")
            except Exception as exc:
                for request in batch:
                    if not request.response_future.done():
                        request.response_future.set_exception(exc)
            else:
                logger.info(
                    "Processed batched inference: batch_size=%d wait_window_ms=%.2f total_ms=%.2f",
                    len(batch),
                    self._batch_timeout_s * 1000,
                    (time.monotonic() - batch_start) * 1000,
                )
                for request, result in zip(batch, results, strict=True):
                    if not request.response_future.done():
                        request.response_future.set_result(result)
            finally:
                for _ in batch:
                    self._request_queue.task_done()

    def _infer_batch(self, obs_batch: list[dict]) -> list[dict]:
        infer_batch = getattr(self._policy, "infer_batch", None)
        if callable(infer_batch):
            return infer_batch(obs_batch)
        return [self._policy.infer(obs) for obs in obs_batch]


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
