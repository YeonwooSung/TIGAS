from flask import Flask


class Server:
    def __init__(
        self,
        app_name: str='tigas',
        port: int=5000,
        debug: bool=True,
        **configs
    ):
        self.name = app_name
        self.port = port
        self.debug = debug
        self.app = Flask(app_name)

    
    def config(self, **configs):
        for config, value in configs:
            self.app.config[config.upper()] = value
    
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

    def run(self):
        self.app.run(host='0.0.0.0', port=self.port, debug=self.debug)
