from fastapi import APIRouter, Request, HTTPException, Form, UploadFile, File
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


@router.post('/i2i', tags=['generate'])
async def i2i(request: Request, text: str = Form(...), image: UploadFile = Form(...)):
    """
    text와 image를 multipart/form-data로 받아서 이미지를 생성하는 API
    """
    # check if the queue is full
    if get_queue_len() >= MAX_LEN:
        raise HTTPException(status_code=429, detail='Queue is full')


    # generate uuid
    uuid_str = str(uuid.uuid4())

    contents = await image.read()
    # save image
    img_path = f'{IMG_DIR_PATH}{uuid_str}_u.png'
    with open(img_path, 'wb') as f:
        f.write(contents)

    # append to queue
    obj = utils.TIGAS_Form(prompt=text, uuid=uuid_str, type='i2i')
    append_to_queue(obj)

    # return uuid
    return {'uuid': uuid_str}


@router.get("/i2i/queue", tags=["generate"])
async def get_size_of_waiting_queue():
    return JSONResponse(content={"num_of_waiting": get_queue_len()})

@router.get("/i2i/queue/{uuid}", tags=["generate"])
async def get_info_of_waiting_queue(uuid: str):
    index = get_object_index_by_uuid()
    if index != -1:
        return JSONResponse(content={"uuid": uuid, "index": index})
    return HTTPException(status_code=400, detail="Not in the waiting list")


@router.get("/i2i/{uuid}/img", tags=["generate"])
async def get_image_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        # if not utils.is_valid_uuid(uuid):
        #     i2i_logger.log(f'/i2i/{uuid} :: error="Invalid UUID"', level='warning')
        #     return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        if not os.path.isfile(f'{IMG_DIR_PATH}{uuid}.png'):
            i2i_logger.log(f'/i2i/{uuid} :: error="Image not found"', level='warning')
            return HTTPException(status_code=404, detail="Image not found")
        
        # return image
        i2i_logger.log(f'/i2i/{uuid} :: success')
        return FileResponse(f'{IMG_DIR_PATH}{uuid}.png')
    except Exception as e:
        i2i_logger.log(f'/i2i/{uuid} :: error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/i2i/{uuid}/status", tags=["generate"])
async def get_status_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        # if not utils.is_valid_uuid(uuid):
        #     i2i_logger.log(f'/i2i/{uuid}/status :: error="Invalid UUID"', level='warning')
        #     return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        if not os.path.isfile(f'{IMG_DIR_PATH}{uuid}.png'):
            i2i_logger.log(f'/i2i/{uuid}/status :: error="Image not found"', level='warning')
            return HTTPException(status_code=404, detail="Image not found")
        
        # return image
        i2i_logger.log(f'/i2i/{uuid}/status :: exists')
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        i2i_logger.log(f'/i2i/{uuid}/status :: error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")
