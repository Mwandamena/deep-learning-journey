import uvicorn
from fastapi import FastAPI
from routes.predict import router as predict_router
from middleware.errors import register_error_handlers
from middleware.logging import LoggingMiddleware

class Config:
    title: str = "MNIST Prediction Server"
    version: str = "v0.0.1"

class App():
    def __init__(self):
        self.server = FastAPI(title=Config.title, version=Config.version)
        self.register_middleware()
        self.register_routes()
        self.register_error_handlers()

    def register_routes(self):
        self.server.include_router(predict_router, prefix="/api/v1")

    def register_middleware(self):
        self.server.add_middleware(LoggingMiddleware)

    def register_error_handlers(self):
        register_error_handlers(self.server)
    
    def start(self):
        uvicorn.run(self.server, host="0.0.0.0", port=8000)


app_instance = App()

app = app_instance.server


if __name__ == "__main__":
    app_instance.start()
