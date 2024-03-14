from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import fastapi as _fapi

import schemas as _schemas
import services as _services
from io import BytesIO
import base64
import traceback


app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to AI Pregnancy Filter API"}


# Endpoint to test the backend
@app.get("/api")
async def root():
    return {"message": "Welcome to the AI Pregnancy Filter with FastAPI"}


@app.post("/api/generate/")
async def generate_image(pregnancyCreate: _schemas.PregnancyCreate = _fapi.Depends()):
    
    try:
        encoded_img = await _services.generate_image(pregnancyCreate=pregnancyCreate)
    except Exception as e:
        print(traceback.format_exc())
        return {"message": f"{e.args}"}
    
    payload = {
        "mime" : "image/jpg",
        "image": encoded_img
        }
    
    return payload
