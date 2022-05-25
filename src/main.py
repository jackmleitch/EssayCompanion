from fastapi import Depends, FastAPI
from pydantic import BaseModel

from .paraphrasing_model.onnx_T5_model import ParaphraseOnnxPipeline

# load paraphrasing model
print('Loading model & tokenizer...')
paraphrasing_pipeline = ParaphraseOnnxPipeline(num_beams=5)
print('Model & tokenizer loaded!')

app = FastAPI()

class ParaphrasingRequest(BaseModel):
    text: str    

class ParaphrasingResponse(BaseModel):
    text: str

@app.get("/")
def get_root():
    return "This is the RESTful API for EssayCompanion"

@app.post("/predict", response_model=ParaphrasingResponse)
async def predict(request: ParaphrasingRequest):
    return ParaphrasingResponse(text=paraphrasing_pipeline(request.text))