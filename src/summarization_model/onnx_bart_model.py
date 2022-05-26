import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from fastBart import get_onnx_model

class SummarizeOnnxPipeline:
    """Model inference pipeline"""

    def __init__(self, num_beams=8) -> None:
        self.model = get_onnx_model("facebook/bart-large-cnn", 
            onnx_models_path="models/summarization/model")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("models/summarization/tokenizer")
        self.num_beams = num_beams
        self.gen_kwargs = {"length_penalty": 0.8, "num_beams": self.num_beams, "max_length": 128}

    def __call__(self, query: str) -> str:
        """
        Summarizes input text using onnx BART model
        :param query: input query to pass to the model
        """
        batch = self.tokenizer(query, max_length=500, truncation=True, return_tensors="pt")
        with torch.no_grad():
            output_tokens = self.model.generate(**batch, **self.gen_kwargs)
        generated_text = self.tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
        return generated_text

if __name__ == '__main__':
    print('Loading BART model + tokenizer...')
    pipe = SummarizeOnnxPipeline()
    print('Loaded!')
    news_article = """
        For a few years, rumors have persisted that Microsoft was exploring building some form 
        of streaming stick to offer Xbox Cloud Gaming via a more affordable dongle, similarly 
        to Chromecast and Google Stadia. The first hint was Project Hobart. More recently, a code 
        name "Keystone" appeared in an Xbox OS list, lending fire to rumors that Microsoft was 
        continuing to explore additional hardware for the Xbox lineup. 

        We can now confirm that that is indeed true, and it pertains to a modernized HDMI 
        streaming device that runs Xbox Game Pass and its cloud gaming service. Microsoft is, 
        however, taking exploring additional iterations of the product before taking it to market. 
        """
    print(pipe(news_article))