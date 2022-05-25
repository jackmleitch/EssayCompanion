python -m transformers.onnx --model=elastic/distilbert-base-uncased-finetuned-conll03-english --feature=token-classification models/ner/model
python src/ner_model/quantize_onnx_model.py