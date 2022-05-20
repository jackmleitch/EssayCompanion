from os import truncate
from fastT5 import (
    OnnxT5,
    export_and_get_onnx_model,
    get_onnx_model,
    get_onnx_runtime_sessions,
    generate_onnx_representation,
    quantize,
)

from transformers import AutoTokenizer

model_ckpt = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
# convert huggingface t5 model to onnx
onnx_model_paths = generate_onnx_representation(model_ckpt)
# quantize the converted model for fast inference and to reduce model size.
quant_model_paths = quantize(onnx_model_paths)
# setup onnx runtime
model_sessions = get_onnx_runtime_sessions(quant_model_paths)
# get the onnx model
model = OnnxT5(model_ckpt, model_sessions)


tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
text = "The ultimate test of your knowledge is your capacity to convey it to another." 

batch = tokenizer(text, truncation=True, padding='longest', max_length=100, return_tensors="pt")
translated = model.generate(**batch, num_beams=5, max_length=100, num_return_sequences=1,
    temperature=1.5)
paraphrased_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
print(paraphrased_text)
