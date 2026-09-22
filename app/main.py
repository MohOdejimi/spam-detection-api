from fastapi import FastAPI
from fastapi.middleware.cors import CORSMddleware

origins = [
    "http://localhost:3000"
]

from app.routes import router

app = FastAPI()
app.add_middleware(
    CORSMddleware, 
    allow_origin = origins,
    allow_credentials = False,
    allow_methods=["*"],
    allow_headers=["*"]
)
app.include_router(router)
