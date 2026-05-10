import asyncio
import logging
from types import SimpleNamespace
from typing import Any, Dict
from uuid import UUID

from fastapi import WebSocket
from starlette.websockets import WebSocketState


Connections = Dict[UUID, Dict[str, Any]]


class ServerFullException(Exception):
    """Exception raised when the server is full."""

    pass


class ConnectionManager:
    def __init__(self):
        self.active_connections: Connections = {}

    async def connect(self, user_id: UUID, websocket: WebSocket, max_queue_size: int = 0):
        await websocket.accept()
        user_count = self.get_user_count()
        print(f"User count: {user_count}")
        if max_queue_size > 0 and user_count >= max_queue_size:
            print("Server is full")
            await websocket.send_json({"status": "error", "message": "Server is full"})
            await websocket.close()
            raise ServerFullException("Server is full")
        print(f"New user connected: {user_id}")
        self.active_connections[user_id] = {
            "websocket": websocket,
            "queue": asyncio.Queue(),
            "debug_info": None,
            "depth_image": None,
            "softedge_image": None,
            "subject_mask_image": None,
            "stylized_image": None,
            "source_image": None,
        }
        await websocket.send_json(
            {"status": "connected", "message": "Connected"},
        )
        await websocket.send_json({"status": "wait"})
        await websocket.send_json({"status": "send_frame"})

    def check_user(self, user_id: UUID) -> bool:
        return user_id in self.active_connections

    async def update_data(self, user_id: UUID, new_data: SimpleNamespace):
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            await queue.put(new_data)

    def clear_queue(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session is None:
            return
        queue = user_session["queue"]
        while not queue.empty():
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    async def get_latest_data(self, user_id: UUID) -> SimpleNamespace:
        user_session = self.active_connections.get(user_id)
        if user_session:
            queue = user_session["queue"]
            try:
                return await queue.get()
            except asyncio.QueueEmpty:
                return None

    def delete_user(self, user_id: UUID):
        user_session = self.active_connections.pop(user_id, None)
        if user_session:
            queue = user_session["queue"]
            while not queue.empty():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    continue

    def get_user_count(self) -> int:
        return len(self.active_connections)

    def get_websocket(self, user_id: UUID) -> WebSocket:
        user_session = self.active_connections.get(user_id)
        if user_session:
            websocket = user_session["websocket"]
            if websocket.client_state == WebSocketState.CONNECTED:
                return user_session["websocket"]
        return None

    def update_outputs(
        self,
        user_id: UUID,
        debug_info: Dict | None = None,
        depth_image: bytes | None = None,
        softedge_image: bytes | None = None,
        subject_mask_image: bytes | None = None,
        stylized_image: bytes | None = None,
        source_image: bytes | None = None,
    ):
        user_session = self.active_connections.get(user_id)
        if user_session is None:
            return
        user_session["debug_info"] = debug_info
        user_session["depth_image"] = depth_image
        user_session["softedge_image"] = softedge_image
        user_session["subject_mask_image"] = subject_mask_image
        user_session["stylized_image"] = stylized_image
        user_session["source_image"] = source_image

    def clear_outputs(self, user_id: UUID):
        self.update_outputs(
            user_id,
            debug_info=None,
            depth_image=None,
            softedge_image=None,
            subject_mask_image=None,
            stylized_image=None,
            source_image=None,
        )

    def get_debug_info(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("debug_info")
        return None

    def get_depth_image(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("depth_image")
        return None

    def get_softedge_image(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("softedge_image")
        return None

    def get_subject_mask_image(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("subject_mask_image")
        return None

    def get_stylized_image(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("stylized_image")
        return None

    def get_source_image(self, user_id: UUID):
        user_session = self.active_connections.get(user_id)
        if user_session:
            return user_session.get("source_image")
        return None

    async def disconnect(self, user_id: UUID):
        websocket = self.get_websocket(user_id)
        if websocket:
            await websocket.close()
        self.delete_user(user_id)

    async def send_json(self, user_id: UUID, data: Dict):
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                await websocket.send_json(data)
        except Exception as e:
            logging.error(f"Error: Send json: {e}")

    async def receive_json(self, user_id: UUID) -> Dict:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_json()
        except Exception as e:
            logging.error(f"Error: Receive json: {e}")

    async def receive_bytes(self, user_id: UUID) -> bytes:
        try:
            websocket = self.get_websocket(user_id)
            if websocket:
                return await websocket.receive_bytes()
        except Exception as e:
            logging.error(f"Error: Receive bytes: {e}")
