from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .generate import router as generate_router

v1 = FastAPI()
v1.include_router(generate_router, prefix="/generate", tags=["generate"])

@v1.get("/")
async def root():
    return RedirectResponse(url="/api/v1/docs")
