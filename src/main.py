from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

from .paraphrasing_model.onnx_T5_model import ParaphraseOnnxPipeline
from .ner_model.onnx_ner_model import NEROnnxModel
from.summarization_model.onnx_bart_model import SummarizeOnnxPipeline

# load T5 paraphrasing model
print('Loading T5 paraphrasing model & tokenizer...')
paraphrasing_pipeline = ParaphraseOnnxPipeline(num_beams=5)
print('T5 paraphrasing model & tokenizer loaded!')
# load distilbert NER model
print('Loading distilbert NER model & tokenizer...')
ner_pipeline = NEROnnxModel()
print('distilbert NER model & tokenizer loaded!')
# loading BART model
print('Loading BART sumarization model & tokenizer...')
summarization_pipeline = SummarizeOnnxPipeline(num_beams=8)
print('BART sumarization model & tokenizer loaded!')

app = FastAPI()

class Request(BaseModel):
    text: str    
class ParagraphResponse(BaseModel):
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

@app.post("/paraphrase", response_model=ParagraphResponse)
async def predict(request: Request):
    return ParagraphResponse(text=paraphrasing_pipeline(request.text))

@app.post("/ner", response_model=NERResponse)
async def predict(request: Request):
    return NERResponse(render_data=ner_pipeline(request.text))

@app.post("/summarize", response_model=ParagraphResponse)
async def predict(request: Request):
    return ParagraphResponse(text=summarization_pipeline(request.text))
