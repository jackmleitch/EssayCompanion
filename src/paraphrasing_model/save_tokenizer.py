from transformers import AutoTokenizer

model_ckpt = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
tokenizer.save_pretrained("models/paraphrase/tokenizer")
