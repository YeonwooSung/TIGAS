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
MAX_LEN = 10
TTI_QUEUE = deque(maxlen=MAX_LEN)

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
            return True
    return False


@router.get("/sample", tags=["generate"], response_class=FileResponse)
async def generate_sample_image():
    sample_uuid = uuid.uuid4()
    sample_text = 'Dogs running on a beach'
    img_name = f'{str(sample_uuid)}.png'
    try:
        obj = utils.TTI_Form(prompt=sample_text, uuid=sample_uuid)
        
        # check if queue is full
        if len(TTI_QUEUE) >= MAX_LEN:
            # check if duplicating prompt exists
            if check_if_prompt_is_in_queue(sample_text):
                wait_until_file_exists(img_name)
                return FileResponse(img_name)
            return HTTPException(status_code=429, detail="Too Many Requests")
        TTI_QUEUE.append(obj)
        
        # forward propagation for inference
        with torch.no_grad():
            sample_image = model(sample_text)

        # clean up the queue
        TTI_QUEUE.remove(obj)

        # convert tensor to pillow image        
        pil_image = convert_model_generate_img_to_pillow_img(sample_image)
        pil_image.save(img_name)
        return FileResponse(img_name)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
    


@router.post("/tti", tags=["generate"], response_class=FileResponse)
async def generate_image_from_text(info : Request):
    try:
        req_info = await info.json()
        if 'text' in req_info:
            sample_uuid = uuid.uuid4()
            text = req_info['text']
            obj = utils.TTI_Form(prompt=text, uuid=sample_uuid)
            img_name = f'{str(sample_uuid)}.png'

            # check if queue is full
            if len(TTI_QUEUE) >= MAX_LEN:
                # check if duplicating prompt exists
                if check_if_prompt_is_in_queue(text):
                    wait_until_file_exists(img_name)
                    return FileResponse(img_name)
                return HTTPException(status_code=429, detail="Too Many Requests")
            TTI_QUEUE.append(obj)

            # forward propagation for inference
            with torch.no_grad():
                image = model(text)
            
            # clean up the queue
            TTI_QUEUE.remove(obj)

            # convert tensor to pillow image
            pil_image = convert_model_generate_img_to_pillow_img(image)
            pil_image.save(img_name)
            return FileResponse(img_name)
        else:
            return HTTPException(status_code=400, detail="Need text to process text-to-image")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
