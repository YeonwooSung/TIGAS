from fastapi import FastAPI
import yaml

# import custom modules
from . import utils
from . import v1
from . import v2
from .signin import router as signin_router


def mount_api_subapplications(app: FastAPI, check_yaml=False) -> None:
    '''
    Mounts all the subapplications to the main application.
    '''
    if check_yaml:
        with open('tigas.yaml') as f:
            parsed_yaml = yaml.safe_load(f)
            apiVersions = parsed_yaml['tigas']['apiVersion']
            if 'v1' in apiVersions:
                app.mount('/api/v1', v1.app)
            if 'v2' in apiVersions:
                app.mount('/api/v2', v2.app)
        return
    app.mount("/api/v1", v1.v1)
    app.mount("/api/v2", v2.v2)
