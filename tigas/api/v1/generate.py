from fastapi import APIRouter
import torch

from .. import utils

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = utils.CustomTextToImageModel(utils.ModelConfig, device, from_pretrained=False)
router = APIRouter()

@router.get("/", tags=["signin"])
async def get_user():
    return {"message": "Hello World"}


@router.post("/", tags=["signin"])
async def create_user():
    return {"message": "Hello World"}
