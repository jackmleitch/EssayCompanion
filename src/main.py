from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict

from .paraphrasing_model.onnx_T5_model import ParaphraseOnnxPipeline
from .ner_model.onnx_ner_model import NEROnnxModel
from .summarization_model.onnx_bart_model import SummarizeOnnxPipeline
from .keyword_model.keyword_extraction import GetKeywords

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

# load KeyBERT model
print('Loading KeyBERT model...')
keyword_pipeline = GetKeywords()
print('KeyBERT model loaded!')

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
class KeywordResponse(BaseModel):
    response: Dict[str, List[str]]
class AllModelsResponse(BaseModel):
    original: str
    paraphrased: ParagraphResponse
    name_entities: NERResponse
    summarized: ParagraphResponse
    keyword_synonyms: KeywordResponse

@app.get("/")
def get_root():
    return "This is the RESTful API for EssayCompanion"

@app.post("/predict", response_model=AllModelsResponse)
async def predict(request: Request):
    paraphrased_text = ParagraphResponse(text=paraphrasing_pipeline(request.text))
    ner_text = NERResponse(render_data=ner_pipeline(request.text))
    if len(request.text) > 200:
        summarized_text = ParagraphResponse(text=summarization_pipeline(request.text))
    else:
        summarized_text = ParagraphResponse(text="Text too short to summarize")
    keyword_synonyms = KeywordResponse(response=
        keyword_pipeline.get_synonyms_for_keywords(request.text))
    return AllModelsResponse(
        original=request.text, paraphrased=paraphrased_text, 
        name_entities=ner_text, summarized=summarized_text,
        keyword_synonyms=keyword_synonyms
    )