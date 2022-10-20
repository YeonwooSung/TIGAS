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
        return HTTPException(status_code=429, detail='Queue is full')

    # validate text and image
    validate_prompt = utils.validate_prompt(text)
    if not validate_prompt:
        i2i_logger.log(f'Invalid text: Non-ascii character is included: text="{text}"', level='warning')
        return HTTPException(status_code=400, detail="Bad Request :: Prompt should be ascii only - no utf-8, no emoji, no special characters.")

    # validate image
    if image.content_type not in ['image/png', 'image/jpeg', 'image/jpg']:
        i2i_logger.log(f'Invalid image: image type is not supported: image="{image.filename}"', level='warning')
        return HTTPException(status_code=400, detail="Bad Request :: Image type is not supported.")

    # get image file extension
    ext = image.filename.split('.')[-1]

    # generate uuid
    uuid_str = str(uuid.uuid4())

    contents = await image.read()
    # save image
    img_path = f'{IMG_DIR_PATH}{uuid_str}_user_upload.{ext}'
    with open(img_path, 'wb') as f:
        f.write(contents)

    # append to queue
    obj = utils.TIGAS_Form(prompt=text, uuid=uuid_str, type='i2i', img_path=img_path)
    append_to_queue(obj)
    queue_len = get_queue_len()

    # register uuid with prompt to the redis server
    utils.register_uuid_with_prompt(text, uuid_str, service='i2i')

    # return uuid and expected time
    return {'uuid': uuid_str, "expected_time": EXPECTED_TIME_PER_IMG * queue_len}


@router.get("/i2i/queue", tags=["generate"])
async def get_size_of_waiting_queue():
    return JSONResponse(content={"num_of_waiting": get_queue_len()})

@router.get("/i2i/queue/{uuid}", tags=["generate"])
async def get_info_of_waiting_queue(uuid: str):
    if not utils.validate_uuid(uuid):
        i2i_logger.log(f'/i2i/{uuid} :: error="Invalid UUID"', level='warning')
        return HTTPException(status_code=400, detail="Bad Request :: Invalid UUID")
    index = get_object_index_by_uuid(uuid)
    if index != -1:
        return JSONResponse(content={"uuid": uuid, "index": index})
    return HTTPException(status_code=400, detail="Not in the waiting list")


@router.get("/i2i/{uuid}/img", tags=["generate"])
async def get_image_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        if not utils.validate_uuid(uuid):
            i2i_logger.log(f'/i2i/{uuid} :: error="Invalid UUID"', level='warning')
            return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        status = utils.get_status(uuid)

        # if status is 'done', return image
        if status == 1:
            if not os.path.isfile(f'{IMG_DIR_PATH}{uuid}.png'):
                i2i_logger.log(f'/i2i/{uuid} :: error="Image file not found"', level='warning')
                return HTTPException(status_code=404, detail="Image file not found")
            i2i_logger.log(f'/i2i/{uuid} :: success')
            return FileResponse(f'{IMG_DIR_PATH}{uuid}.png')

        # if status is 'error', return error message
        elif status == -1:
            i2i_logger.log(f'/i2i/{uuid} :: error="status=-1"', level='error')
            return HTTPException(status_code=500, detail="Error occurred while generating image")
        return HTTPException(status_code=404, detail="Image not found")
    except Exception as e:
        i2i_logger.log(f'/i2i/{uuid} :: error="{e}"', level='error')
        return HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/i2i/{uuid}/status", tags=["generate"])
async def get_status_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        if not utils.validate_uuid(uuid):
            i2i_logger.log(f'/i2i/{uuid}/status :: error="Invalid UUID"', level='warning')
            return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        status = utils.get_status(uuid)

        # if status is 'done', return image
        if status == 1:
            i2i_logger.log(f'/i2i/{uuid}/status :: success')
            return JSONResponse(content={"status": "done"})
        
        # if status is 'error', return error message
        elif status == -1:
            i2i_logger.log(f'/i2i/{uuid}/status :: error="status=-1"', level='error')
            return HTTPException(status_code=500, detail="Error occurred while generating image")
        
        i2i_logger.log(f'/i2i/{uuid}/status :: pending', level='debug')
        return HTTPException(status_code=400, detail="Status pending")
    except Exception as e:
        i2i_logger.log(f'/i2i/{uuid}/status :: error="{e}"', level='error')
        return HTTPException(status_code=500, detail="Internal Server Error")
