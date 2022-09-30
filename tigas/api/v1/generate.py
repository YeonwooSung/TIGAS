import base64
from fastapi import APIRouter
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from io import BytesIO
import torch
from PIL import Image


from .. import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
router = APIRouter()

#TODO write codes to generate image


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
async def get_user():
    sample_text = 'Dogs running on a beach'
    sample_image = model(sample_text)
    pil_image = convert_model_generate_img_to_pillow_img(sample_image)
    return StreamingResponse(BytesIO(pil_image), media_type="image/png")
    # img_converted = from_image_to_bytes(pil_image)
    # return JSONResponse([img_converted])


@router.post("/", tags=["generate"])
async def create_user():
    return {"message": "Hello World"}
