from transformers import AutoTokenizer

model_ckpt = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.save_pretrained("models/summarization/tokenizer")
