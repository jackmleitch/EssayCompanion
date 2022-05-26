from fastBart import export_and_get_onnx_model
from transformers import AutoTokenizer

model_ckpt = 'facebook/bart-large-cnn'
output_path = "models/summarization/model"
model = export_and_get_onnx_model(model_ckpt, output_path)