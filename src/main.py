from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .paraphrasing_model.onnx_T5_model import ParaphraseOnnxPipeline
from .ner_model.onnx_ner_model import NEROnnxModel

# load paraphrasing model
print('Loading paraphrasing model & tokenizer...')
paraphrasing_pipeline = ParaphraseOnnxPipeline(num_beams=5)
print('Paraphrasing model & tokenizer loaded!')
# load NER model
print('Loading NER model & tokenizer...')
ner_pipeline = NEROnnxModel()
print('NER model & tokenizer loaded!')

app = FastAPI()

class Request(BaseModel):
    text: str    
class ParaphrasingResponse(BaseModel):
    text: str

class EntityData(BaseModel):
    start: int
    end: int
    label: str
class RenderData(BaseModel):
    text: str
    ents: List[EntityData]
    title: None
class NERResponse(BaseModel):
    render_data: List[RenderData]


@app.get("/")
def get_root():
    return "This is the RESTful API for EssayCompanion"

@app.post("/paraphrase", response_model=ParaphrasingResponse)
async def predict(request: Request):
    return ParaphrasingResponse(text=paraphrasing_pipeline(request.text))

@app.post("/ner", response_model=NERResponse)
async def predict(request: Request):
    return NERResponse(render_data=ner_pipeline(request.text))
