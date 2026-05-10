import logging
import mimetypes
import os
import time
import uuid
from types import SimpleNamespace

import markdown2
import torch
from config import Args, config
from connection_manager import ConnectionManager, ServerFullException
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from util import bytes_to_pil, pil_to_bytes, pil_to_frame
from vid2vid import Pipeline


# fix mime error on windows
mimetypes.add_type("application/javascript", ".js")

THROTTLE = 1.0 / 120
# logging.basicConfig(level=logging.DEBUG)


class App:
    def __init__(self, config: Args):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = torch.float16
        pipeline = Pipeline(config, device, torch_dtype)
        self.args = config
        self.pipeline = pipeline
        self.app = FastAPI()
        self.conn_manager = ConnectionManager()
        self.init_app()

    def init_app(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @self.app.websocket("/api/ws/{user_id}")
        async def websocket_endpoint(user_id: uuid.UUID, websocket: WebSocket):
            try:
                await self.conn_manager.connect(user_id, websocket, self.args.max_queue_size)
                await handle_websocket_data(user_id)
            except ServerFullException as e:
                logging.error(f"Server Full: {e}")
            finally:
                await self.conn_manager.disconnect(user_id)
                logging.info(f"User disconnected: {user_id}")

        async def handle_websocket_data(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                return HTTPException(status_code=404, detail="User not found")
            last_time = time.time()
            try:
                while True:
                    if self.args.timeout > 0 and time.time() - last_time > self.args.timeout:
                        await self.conn_manager.send_json(
                            user_id,
                            {
                                "status": "timeout",
                                "message": "Your session has ended",
                            },
                        )
                        return
                    data = await self.conn_manager.receive_json(user_id)
                    if not isinstance(data, dict):
                        return
                    if data["status"] == "next_frame":
                        info = self.pipeline.Info()
                        params = await self.conn_manager.receive_json(user_id)
                        params = self.pipeline.InputParams(**params)
                        params = SimpleNamespace(**params.model_dump())
                        if info.input_mode == "image":
                            image_data = await self.conn_manager.receive_bytes(user_id)
                            if len(image_data) == 0:
                                await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                                continue
                            params.image = bytes_to_pil(image_data)
                        await self.conn_manager.update_data(user_id, params)

            except Exception as e:
                logging.error(f"Websocket Error: {e}, {user_id} ")

        @self.app.get("/api/queue")
        async def get_queue_size():
            queue_size = self.conn_manager.get_user_count()
            return JSONResponse({"queue_size": queue_size})

        @self.app.get("/api/stream/{user_id}")
        async def stream(user_id: uuid.UUID, request: Request):
            try:
                async def generate():
                    while True:
                        if await request.is_disconnected() or not self.conn_manager.check_user(user_id):
                            return
                        last_time = time.time()
                        await self.conn_manager.send_json(user_id, {"status": "send_frame"})
                        params = await self.conn_manager.get_latest_data(user_id)
                        if params is None:
                            continue
                        image, debug_info, depth_image, softedge_image, subject_mask_image, stylized_image = (
                            self.pipeline.predict(params)
                        )
                        if image is None:
                            continue
                        src = getattr(params, "image", None)
                        source_bytes = None
                        if src is not None:
                            src_rgb = src.convert("RGB") if src.mode != "RGB" else src
                            source_bytes = pil_to_bytes(src_rgb, "JPEG")
                        depth_bytes = pil_to_bytes(depth_image) if depth_image is not None else None
                        softedge_bytes = pil_to_bytes(softedge_image) if softedge_image is not None else None
                        subject_mask_bytes = pil_to_bytes(subject_mask_image) if subject_mask_image is not None else None
                        stylized_bytes = pil_to_bytes(stylized_image) if stylized_image is not None else None
                        merged_info = {
                            **debug_info,
                            "has_source_preview": source_bytes is not None,
                            "has_softedge_preview": softedge_bytes is not None,
                            "has_subject_mask_preview": subject_mask_bytes is not None,
                            "has_stylized_preview": stylized_bytes is not None,
                        }
                        self.conn_manager.update_outputs(
                            user_id,
                            debug_info=merged_info,
                            depth_image=depth_bytes,
                            softedge_image=softedge_bytes,
                            subject_mask_image=subject_mask_bytes,
                            stylized_image=stylized_bytes,
                            source_image=source_bytes,
                        )
                        frame = pil_to_frame(image)
                        yield frame
                        if self.args.debug:
                            print(f"Time taken: {time.time() - last_time}")

                return StreamingResponse(
                    generate(),
                    media_type="multipart/x-mixed-replace;boundary=frame",
                    headers={"Cache-Control": "no-cache"},
                )
            except Exception as e:
                logging.error(f"Streaming Error: {e}, {user_id} ")
                return HTTPException(status_code=404, detail="User not found")

        # route to setup frontend
        @self.app.get("/api/settings")
        async def settings():
            info_schema = self.pipeline.Info.model_json_schema()
            info = self.pipeline.Info()
            if info.page_content:
                page_content = markdown2.markdown(info.page_content)

            input_params = self.pipeline.InputParams.model_json_schema()
            return JSONResponse(
                {
                    "info": info_schema,
                    "input_params": input_params,
                    "max_queue_size": self.args.max_queue_size,
                    "page_content": page_content if info.page_content else "",
                }
            )

        @self.app.get("/api/debug/{user_id}")
        async def get_debug_info(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            return JSONResponse(self.conn_manager.get_debug_info(user_id) or {})

        @self.app.post("/api/reset/{user_id}")
        async def reset_runtime_state(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            self.conn_manager.clear_queue(user_id)
            self.conn_manager.clear_outputs(user_id)
            self.pipeline.reset_runtime_state()
            return JSONResponse({"status": "ok"})

        @self.app.get("/api/depth/{user_id}")
        async def get_depth_image(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            depth_image = self.conn_manager.get_depth_image(user_id)
            if depth_image is None:
                return Response(status_code=204)
            return Response(
                content=depth_image,
                media_type="image/png",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @self.app.get("/api/softedge/{user_id}")
        async def get_softedge_image(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            softedge_image = self.conn_manager.get_softedge_image(user_id)
            if softedge_image is None:
                return Response(status_code=204)
            return Response(
                content=softedge_image,
                media_type="image/png",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @self.app.get("/api/subject-mask/{user_id}")
        async def get_subject_mask_image(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            subject_mask_image = self.conn_manager.get_subject_mask_image(user_id)
            if subject_mask_image is None:
                return Response(status_code=204)
            return Response(
                content=subject_mask_image,
                media_type="image/png",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @self.app.get("/api/stylized/{user_id}")
        async def get_stylized_image(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            stylized_image = self.conn_manager.get_stylized_image(user_id)
            if stylized_image is None:
                return Response(status_code=204)
            return Response(
                content=stylized_image,
                media_type="image/png",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        @self.app.get("/api/preview/input/{user_id}")
        async def get_input_preview(user_id: uuid.UUID):
            if not self.conn_manager.check_user(user_id):
                raise HTTPException(status_code=404, detail="User not found")
            source_image = self.conn_manager.get_source_image(user_id)
            if source_image is None:
                return Response(status_code=204)
            return Response(
                content=source_image,
                media_type="image/jpeg",
                headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
            )

        if not os.path.exists("public"):
            os.makedirs("public")

        self.app.mount("/", StaticFiles(directory="./frontend/public", html=True), name="public")


app = App(config).app
