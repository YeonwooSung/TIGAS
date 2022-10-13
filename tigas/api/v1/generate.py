import base64
from collections import deque
from genericpath import isfile
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from io import BytesIO
import os
import time
import torch
import uuid
from PIL import Image


from .. import utils


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
model.eval()

# ------------------------------------
# constants
LOG_DIR_PATH = '/home/ys60/logs/'
IMG_DIR_PATH = '/home/ys60/images/'
MAX_LEN = 10
# ------------------------------------

# check if essential directories exist
if not os.path.isdir(IMG_DIR_PATH):
    os.mkdir(IMG_DIR_PATH)
if not os.path.isdir(LOG_DIR_PATH):
    os.mkdir(LOG_DIR_PATH)

TTI_QUEUE = deque(maxlen=MAX_LEN)

tti_logger = utils.StableLogger(f'{LOG_DIR_PATH}tti.log', name='tti_logger')

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

def convert_model_generate_img_to_pillow_img(image):
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images[0]


def wait_until_file_exists(path:str):
    while not os.path.isfile(path):
        time.sleep(10)

def check_if_prompt_is_in_queue(prompt):
    for i in range(len(TTI_QUEUE)):
        if TTI_QUEUE[i].prompt == prompt:
            return True, TTI_QUEUE[i].uuid
    return False, None


@router.get("/sample", tags=["generate"], response_class=FileResponse)
async def generate_sample_image():
    sample_uuid = uuid.uuid4()
    sample_text = 'Dogs running on a beach'
    try:
        obj = utils.TTI_Form(prompt=sample_text, uuid=sample_uuid)
        
        # check if queue is full
        if len(TTI_QUEUE) >= MAX_LEN:
            # check if duplicating prompt exists
            has_duplicating, target_uuid = check_if_prompt_is_in_queue(sample_text)
            if has_duplicating:
                tti_logger.log(f'Found duplicating prompt: {sample_text} -> original uuid: {target_uuid}, requested uuid: {sample_uuid}')
                img_name = f'{IMG_DIR_PATH}{target_uuid}.png'
                wait_until_file_exists(img_name)
                return FileResponse(img_name)
            tti_logger.log(f'Queue is full.. Block new request: prompt="{sample_text}" uuid="{sample_uuid}"', level='warn')
            return HTTPException(status_code=429, detail="Too Many Requests")
        TTI_QUEUE.append(obj)
        tti_logger.log(f'/sample :: uuid="{sample_uuid}", prompt="{sample_text}"')
        
        # forward propagation for inference
        with torch.no_grad():
            sample_image = model(sample_text)

        # clean up the queue
        TTI_QUEUE.remove(obj)

        # convert tensor to pillow image        
        pil_image = convert_model_generate_img_to_pillow_img(sample_image)
        img_name = f'{IMG_DIR_PATH}{str(sample_uuid)}.png'
        tti_logger.log(f'/sample :: uuid="{sample_uuid}", img_name="{img_name}"')
        pil_image.save(img_name)
        return FileResponse(img_name)
    except Exception as e:
        tti_logger.log(f'/sample :: uuid="{sample_uuid}", error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


@router.post("/tti", tags=["generate"], response_class=FileResponse)
async def generate_image_from_text(info : Request):
    try:
        req_info = await info.json()
        if 'text' in req_info:
            sample_uuid = uuid.uuid4()
            text = req_info['text']
            obj = utils.TTI_Form(prompt=text, uuid=sample_uuid)

            # check if queue is full
            if len(TTI_QUEUE) >= MAX_LEN:
                tti_logger.log(f'Queue is full.. Block new request: prompt="{text}" uuid="{sample_uuid}"', level='warn')
                return HTTPException(status_code=429, detail="Too Many Requests")
            TTI_QUEUE.append(obj)
            tti_logger.log(f'/tti :: uuid="{sample_uuid}", prompt="{text}"')

            # forward propagation for inference
            with torch.no_grad():
                image = model(text)
            
            # clean up the queue
            TTI_QUEUE.remove(obj)

            # convert tensor to pillow image
            pil_image = convert_model_generate_img_to_pillow_img(image)
            img_name = f'{IMG_DIR_PATH}{str(sample_uuid)}.png'
            pil_image.save(img_name)
            tti_logger.log(f'/tti :: uuid="{sample_uuid}", img_name="{img_name}"')
            return FileResponse(img_name)
        else:
            tti_logger.log(f'/tti :: error="No text in request"', level='warn')
            return HTTPException(status_code=400, detail="Need text to process text-to-image")
    except Exception as e:
        tti_logger.log(f'/tti :: uuid="{sample_uuid}", error="{e}"', level='error')
        raise HTTPException(status_code=500, detail="Internal Server Error")
