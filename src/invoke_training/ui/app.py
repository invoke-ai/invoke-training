from pathlib import Path

import gradio as gr
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from invoke_training.ui.pages.data_page import DataPage
from invoke_training.ui.pages.training_page import TrainingPage
from invoke_training.ui.pages.model_merge_page import ModelMergePage


def build_app():
    training_page = TrainingPage()
    model_merge_page = ModelMergePage()
    data_page = DataPage()

    app = FastAPI()

    @app.get("/")
    async def root():
        index_path = Path(__file__).parent / "index.html"
        return FileResponse(index_path)

    app.mount("/assets", StaticFiles(directory=Path(__file__).parent.parent / "assets"), name="assets")

    app = gr.mount_gradio_app(app, training_page.app(), "/train", app_kwargs={"favicon_path": "/assets/favicon.png"})
    app = gr.mount_gradio_app(app, data_page.app(), "/data", app_kwargs={"favicon_path": "/assets/favicon.png"})
    app = gr.mount_gradio_app(app, model_merge_page.app(), "/merge", app_kwargs={"favicon_path": "/assets/favicon.png"})
    return app
