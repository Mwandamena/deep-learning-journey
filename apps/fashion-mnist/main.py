import uvicorn
from fastapi import FastAPI
from routes.predict import router as predict_router
from api_middleware import register_error_handlers, LoggingMiddleware


class Config:
    title: str = "FashionMNIST Prediction Server"
    version: str = "v0.0.1"


class App():
    def __init__(self):
        self.server = FastAPI(title=Config.title, version=Config.version)
        self.register_middleware()
        self.register_routes()
        self.setup_error_handlers()

    def register_routes(self):
        self.server.include_router(predict_router, prefix="/api/v1")

    def register_middleware(self):
        self.server.add_middleware(LoggingMiddleware, logger_name="fashion-mnist")

    def setup_error_handlers(self):
        register_error_handlers(self.server, logger_name="fashion-mnist")

    def start(self):
        uvicorn.run(self.server, host="0.0.0.0", port=8001)


app_instance = App()
app = app_instance.server

if __name__ == "__main__":
    app_instance.start()