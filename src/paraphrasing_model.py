from ast import Str
import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from nltk.tokenize import sent_tokenize

# download nltk punkt sentence tokenizer if it's not found in files
try:
    nltk_path = nltk.find("corpora/punkt")
except Exception:
    nltk.download('punkt')

class ParaphraseModel: 
    """
    Provides utility to load HuggingFace paraphrasing model and generate paraphrased text.
    """
    def __init__(self, model_ckpt="tuner007/pegasus_paraphrase", num_beams=10) -> None:
        """
        :param model_ckpt: path to HuggingFace model checkpoint, default is the PEGASUS paraphraser
        :param num_beams: number of beams to perform beam search with when generating new text
        """
        self.num_beams = num_beams
        self.torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading {model_ckpt} tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
        print("Model loaded!")

    def paraphrase_text(self, input_text: str) -> str:
        """
        Tokenize sentences and then encoder-decoder model to paraphrase text.
        :param input_text: input text to feed to model
        :return: paraphrased text
        """
        sentences = sent_tokenize(input_text)
        batch = (self.tokenizer(sentences, truncation=True, padding="longest",
            max_length=100, return_tensors="pt").to(self.torch_device))
        translated = self.model.generate(**batch, max_length=60, num_beams=self.num_beams,
            num_return_sequences=1, temperature=1.5)
        paraphrased_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        paraphrased_text = " ".join(paraphrased_text)
        return paraphrased_text

if __name__ == "__main__":
    text = """
        The ultimate test of your knowledge is your capacity to convey it to another. 
        Best way: learning by teaching & practicing. Heard D. Mermin once describe a
        person who has a good understanding of physics but can't convey it, as being 
        a natural phenomenon, not a physicist. 
        """
    paraphraser = ParaphraseModel()
    paraphased_text = paraphraser.paraphrase_text(text)
    print(paraphased_text)
