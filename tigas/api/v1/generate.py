import base64
from fastapi import APIRouter, Request, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, JSONResponse
from io import BytesIO
import os
import uuid
import yaml


# custom module
from .. import utils
from ..utils import append_to_queue, get_queue_len, get_object_index_by_uuid


# parse config file
with open('tigas.yaml', 'r') as f:
    config = yaml.safe_load(f)
    tti_generate_config = config['tigas']['generate']['tti']
    _MAX_SIZE = config['tigas']['generate']['max']
    path_config = tti_generate_config['path']
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
tti_logger = utils.StableLogger(f'{LOG_DIR_PATH}tti.log', name='tti_logger')

# API router for /api/v1/generate
router = APIRouter()



def from_image_to_bytes(img):
    """
    pillow image 객체를 bytes로 변환
    """
    # Pillow 이미지 객체를 Bytes로 변환
    imgByteArr = BytesIO()
    img.save(imgByteArr, format=img.format)
    imgByteArr = imgByteArr.getvalue()
    # Base64로 Bytes를 인코딩
    encoded = base64.b64encode(imgByteArr)
    # Base64로 ascii로 디코딩
    decoded = encoded.decode('ascii')
    return decoded


@router.get("/sample", tags=["generate"])
async def generate_sample_image():
    sample_uuid = uuid.uuid4()
    sample_text = 'Dogs running on a beach'
    try:
        obj = utils.TIGAS_Form(prompt=sample_text, uuid=sample_uuid)

        len_of_queue = get_queue_len()

        # check if queue is full
        if len_of_queue >= MAX_LEN:
            tti_logger.log(f'Queue is full.. Block new request: prompt="{sample_text}" uuid="{sample_uuid}"', level='warning')
            return HTTPException(status_code=429, detail="Too Many Requests")

        # append to request form to the queue
        append_to_queue(obj)
        tti_logger.log(f'/sample :: uuid="{sample_uuid}", prompt="{sample_text}"')

        # update the len_of_queue since we appended the new obj
        len_of_queue = get_queue_len()

        response_obj = {
            "uuid": sample_uuid,
            "prompt": sample_text,
            "queue": len_of_queue,
            "expected_time": len_of_queue * EXPECTED_TIME_PER_IMG,
        }
        json_compatible_item_data = jsonable_encoder(response_obj)
        return JSONResponse(content=json_compatible_item_data)
    except Exception as e:
        tti_logger.log(f'/sample :: uuid="{sample_uuid}", error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


@router.post("/tti", tags=["generate"])
async def generate_image_from_text(info : Request):
    try:
        req_info = await info.json()
        if 'text' in req_info:
            sample_uuid = uuid.uuid4()
            text = req_info['text']
            obj = utils.TIGAS_Form(prompt=text, uuid=sample_uuid)

            len_of_queue = get_queue_len()

            # check if queue is full
            if len_of_queue >= MAX_LEN:
                tti_logger.log(f'Queue is full.. Block new request: prompt="{text}" uuid="{sample_uuid}"', level='warning')
                return HTTPException(status_code=429, detail="Too Many Requests")
            
            # append user info to the waiting queue
            append_to_queue(obj)
            # log the request
            tti_logger.log(f'/tti :: uuid="{sample_uuid}", prompt="{text}"')

            # update the len_of_queue since we appended the new obj
            len_of_queue = get_queue_len()

            response_obj = {
                "uuid": sample_uuid,
                "prompt": text,
                "queue": len_of_queue,
                "expected_time": len_of_queue * EXPECTED_TIME_PER_IMG,
            }
            json_compatible_item_data = jsonable_encoder(response_obj)
            return JSONResponse(content=json_compatible_item_data)
        else:
            tti_logger.log(f'/tti :: error="No text in request"', level='warning')
            return HTTPException(status_code=400, detail="Need text to process text-to-image")
    except Exception as e:
        tti_logger.log(f'/tti :: uuid="{sample_uuid}", error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/tti/queue", tags=["generate"])
async def get_size_of_waiting_queue():
    return JSONResponse(content={"num_of_waiting": get_queue_len()})

@router.get("/tti/queue/{uuid}", tags=["generate"])
async def get_info_of_waiting_queue(uuid: str):
    index = get_object_index_by_uuid()
    if index != -1:
        return JSONResponse(content={"uuid": uuid, "index": index})
    return HTTPException(status_code=400, detail="Not in the waiting list")


@router.get("/tti/{uuid}/img", tags=["generate"])
async def get_image_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        # if not utils.is_valid_uuid(uuid):
        #     tti_logger.log(f'/tti/{uuid} :: error="Invalid UUID"', level='warning')
        #     return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        if not os.path.isfile(f'{IMG_DIR_PATH}{uuid}.png'):
            tti_logger.log(f'/tti/{uuid} :: error="Image not found"', level='warning')
            return HTTPException(status_code=404, detail="Image not found")
        
        # return image
        tti_logger.log(f'/tti/{uuid} :: success')
        return FileResponse(f'{IMG_DIR_PATH}{uuid}.png')
    except Exception as e:
        tti_logger.log(f'/tti/{uuid} :: error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.get("/tti/{uuid}/status", tags=["generate"])
async def get_status_from_uuid(uuid: str):
    try:
        # check if uuid is valid
        # if not utils.is_valid_uuid(uuid):
        #     tti_logger.log(f'/tti/{uuid}/status :: error="Invalid UUID"', level='warning')
        #     return HTTPException(status_code=400, detail="Invalid UUID")
        
        # check if image exists
        if not os.path.isfile(f'{IMG_DIR_PATH}{uuid}.png'):
            tti_logger.log(f'/tti/{uuid}/status :: error="Image not found"', level='warning')
            return HTTPException(status_code=404, detail="Image not found")
        
        # return image
        tti_logger.log(f'/tti/{uuid}/status :: exists')
        return JSONResponse(content={"status": "ok"})
    except Exception as e:
        tti_logger.log(f'/tti/{uuid}/status :: error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")
