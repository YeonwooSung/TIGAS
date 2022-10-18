from fastapi import APIRouter, Request, HTTPException, Form
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
import os
import uuid
import yaml


# custom module
from .. import utils
from ..utils import append_to_queue, get_queue_len, get_object_index_by_uuid


# parse config file
with open('tigas.yaml', 'r') as f:
    config = yaml.safe_load(f)
    generate_config = config['tigas']['generate']
    _MAX_SIZE = generate_config['max']
    path_config = generate_config['i2i']['path']
    _LOG_DIR_PATH = path_config['log'] if 'log' in path_config else '/home/ys60/logs/'
    _IMG_DIR_PATH = path_config['img'] if 'img' in path_config else '/home/ys60/images/'


# ------------------------------------
# constants
LOG_DIR_PATH = _LOG_DIR_PATH
IMG_DIR_PATH = _IMG_DIR_PATH
MAX_LEN = _MAX_SIZE

EXPECTED_TIME_PER_IMG = 31
# ------------------------------------

# check if essential directories exist
if not os.path.isdir(IMG_DIR_PATH):
    os.mkdir(IMG_DIR_PATH)
if not os.path.isdir(LOG_DIR_PATH):
    os.mkdir(LOG_DIR_PATH)

# python logging.Logger based custom logger
i2i_logger = utils.StableLogger(f'{LOG_DIR_PATH}i2i.log', name='i2i_logger')

# API router for /api/v2/generate
router = APIRouter()


@router.post('/i2i', tags=['generate', 'i2i'])
async def i2i(request: Request, text: str = Form(...), image: bytes = Form(...)):
    """
    text와 image를 multipart/form-data로 받아서 이미지를 생성하는 API
    """
    #TODO ??
    # check if the queue is full
    if get_queue_len() >= MAX_LEN:
        raise HTTPException(status_code=429, detail='Queue is full')

    # generate uuid
    uuid_str = str(uuid.uuid4())

    # save image
    img_path = f'{IMG_DIR_PATH}{uuid_str}.jpg'
    with open(img_path, 'wb') as f:
        f.write(image)

    # append to queue
    obj = utils.TIGAS_Form(prompt=text, uuid=uuid_str, type='i2i')
    append_to_queue(obj)

    # return uuid
    return {'uuid': uuid_str}
