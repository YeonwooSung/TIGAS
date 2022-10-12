import base64
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from io import BytesIO
import torch
import uuid
from PIL import Image


from .. import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
model.eval()
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


@router.get("/sample", tags=["generate"])
async def generate_sample_image():
    sample_uuid = uuid.uuid4()
    sample_text = 'Dogs running on a beach'
    with torch.no_grad():
        sample_image = model(sample_text)
    pil_image = convert_model_generate_img_to_pillow_img(sample_image)
    img_name = f'{str(sample_uuid)}.png'
    pil_image.save(img_name)
    return FileResponse(img_name)
    #return StreamingResponse(BytesIO(pil_image), media_type="image/png")
    # img_converted = from_image_to_bytes(pil_image)
    # return JSONResponse([img_converted])


@router.post("/tti", tags=["generate"])
async def generate_image_from_text(info : Request):
    req_info = await info.json()
    if 'text' in req_info:
        sample_uuid = uuid.uuid4()
        text = req_info['text']
        with torch.no_grad():
            image = model(text)
        pil_image = convert_model_generate_img_to_pillow_img(image)
        img_name = f'{str(sample_uuid)}.png'
        pil_image.save(img_name)
        return FileResponse(img_name)
    return HTTPException(status_code=400, detail="Need text to process text-to-image")
    #return {"message": "Hello World"}

