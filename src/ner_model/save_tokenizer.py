from transformers import AutoTokenizer

model_ckpt = "elastic/distilbert-base-uncased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.save_pretrained("models/ner/tokenizer")
