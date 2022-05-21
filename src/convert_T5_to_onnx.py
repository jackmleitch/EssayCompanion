import nltk
from fastT5 import export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize

# download nltk punkt sentence tokenizer if it's not found in files
try:
    nltk_path = nltk.find("corpora/punkt")
except Exception:
    nltk.download('punkt')

model_ckpt = "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"

# export model to ONNX and quantize from 32bit to 8bit
print(f"Exporting f{model_ckpt} to ONNX and quantizing...")
model = export_and_get_onnx_model(model_ckpt)
print("Model exported and saved to /models directory")

if __name__ == '__main__':
    # run already exported model
    model = get_onnx_model(model_ckpt)
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
    text = """
        The ultimate test of your knowledge is your capacity to convey it to another. 
        The best way of learning is by teaching & practicing. D. Mermin once described a
        person who has a good understanding of physics but can't convey it, as being 
        a natural phenomenon, not a physicist. 
        """    
    sentences = sent_tokenize(text)
    paraphrased_text = []
    batch = (tokenizer(sentences, truncation=True, padding="longest",
        max_length=100, return_tensors="pt").to('cpu'))
    translated = model.generate(**batch, max_length=100, num_beams=5,
        num_return_sequences=1, temperature=1.5)
    paraphrased_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    if model_ckpt == "ramsrigouthamg/t5-large-paraphraser-diverse-high-quality":
        # remove 'paraphrasedoutput: ' from result
        paraphrased_text = [sentence[19:] for sentence in paraphrased_text]
    paraphrased_text = " ".join(paraphrased_text)
    print(paraphrased_text)