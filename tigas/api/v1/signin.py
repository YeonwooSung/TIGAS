from fastapi import APIRouter
from fastapi.responses import RedirectResponse


router = APIRouter()

@router.get("/", tags=["signin"])
async def get_user():
    return RedirectResponse(url="/signin")
