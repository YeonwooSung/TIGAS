from fastapi import FastAPI, APIRouter, Request
from fastapi.responses import RedirectResponse

from api import mount_api_subapplications
from api.utils import private



class Server:
    def __init__(
        self,
        app_name: str='tigas',
        **configs
    ):
        self.name = app_name
        self.app = FastAPI()
        self.mount_api_versions()
        self.add_index()

    @private
    def mount_api_versions(self):
        mount_api_subapplications(self.app)
    
    @private
    def add_index(self):
        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/api/v1/docs")

    def add_endpoint(
        self,
        endpoint:str=None, 
        endpoint_name:str=None, 
        handler:APIRouter=None, 
    ):
        if endpoint is None:
            if endpoint_name is None:
                raise ValueError('Endpoint name is required')
            endpoint = f"/{endpoint_name}"
        if handler is None:
            raise ValueError('Handler is required')
        self.app.include_router(handler, prefix=endpoint, tags=[endpoint_name])
