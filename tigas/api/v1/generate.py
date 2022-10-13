import base64
from collections import deque
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from io import BytesIO
import torch
import uuid
from PIL import Image


from .. import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
model.eval()
TTI_QUEUE = deque(maxlen=10)

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


@router.get("/sample", tags=["generate"], response_class=FileResponse)
async def generate_sample_image():
    sample_uuid = uuid.uuid4()
    sample_text = 'Dogs running on a beach'
    try:
        obj = utils.TTI_Form(prompt=sample_text, uuid=sample_uuid)
        
        if len(TTI_QUEUE) >= TTI_QUEUE.maxlen:
            #TODO: duplicate prompt -> wait until the target image is generated
            return HTTPException(status_code=429, detail="Too Many Requests")
        TTI_QUEUE.append(obj)
        
        # forward propagation for inference
        with torch.no_grad():
            sample_image = model(sample_text)

        # clean up the queue
        TTI_QUEUE.remove(obj)

        # convert tensor to pillow image        
        pil_image = convert_model_generate_img_to_pillow_img(sample_image)
        img_name = f'{str(sample_uuid)}.png'
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

            if len(TTI_QUEUE) >= TTI_QUEUE.maxlen:
                #TODO: duplicate prompt -> wait until the target image is generated
                return HTTPException(status_code=429, detail="Too Many Requests")
            TTI_QUEUE.append(obj)

            # forward propagation for inference
            with torch.no_grad():
                image = model(text)
            
            # clean up the queue
            TTI_QUEUE.remove(obj)

            # convert tensor to pillow image
            pil_image = convert_model_generate_img_to_pillow_img(image)
            img_name = f'{str(sample_uuid)}.png'
            pil_image.save(img_name)
            return FileResponse(img_name)
        else:
            return HTTPException(status_code=400, detail="Need text to process text-to-image")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Internal Server Error")
