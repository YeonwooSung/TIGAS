from fastapi import APIRouter


router = APIRouter()

@router.get("/", tags=["users"])
async def get_user():
    return {"message": "Hello World"}


@router.post("/", tags=["users"])
async def create_user():
    return {"message": "Hello World"}
