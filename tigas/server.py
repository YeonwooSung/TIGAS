from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse


class Server:
    def __init__(
        self,
        app_name: str='tigas',
        **configs
    ):
        self.name = app_name
        self.app = FastAPI()


    def add_endpoint(
        self,
        endpoint=None, 
        endpoint_name=None, 
        handler=None, 
        methods=['GET'], 
        *args, 
        **kwargs
    ):
        self.app.add_url_rule(endpoint, endpoint_name, handler, methods=methods, *args, **kwargs)
