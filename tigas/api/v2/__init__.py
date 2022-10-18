from fastapi import FastAPI
from fastapi.responses import RedirectResponse

# custom module
from .generate import router as generate_router

v2 = FastAPI()
v2.include_router(generate_router, prefix="/generate", tags=["generate"])

@v2.get("/")
async def root():
    return RedirectResponse(url="/api/v2/docs")
