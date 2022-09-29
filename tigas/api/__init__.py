
from fastapi import FastAPI

from . import utils
from . import v1
from .signin import router as signin_router


def mount_api_subapplications(app: FastAPI) -> None:
    '''
    Mounts all the subapplications to the main application.
    '''
    app.mount("/api/v1", v1.v1)
