import configparser
import requests
import nltk
from nltk.tokenize import sent_tokenize

# download nltk punkt sentence tokenizer if it's not found in files
try:
    nltk_path = nltk.find("corpora/punkt")
except Exception:
    nltk.download('punkt')

def query(api_key: str, payload: dict) -> dict:
    """
    Queries the HuggingFace hosted inference API to get the response of the Pegasus Model
    :param api_key: secret key to query HuggingFace API 
    :param payload: text payload to post to API
    :return: response payload from API containing paraphrased text
    """
    API_URL = "https://api-inference.huggingface.co/models/tuner007/pegasus_paraphrase"
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def get_paraphrases(api_key: str, text: str) -> str: 
    """
    Precesses input text and returns respective paraphrased text
    :param api_key: secret key to query HuggingFace API
    :param text: input text to paraphrase
    :return: paraprased version of text
    """
    sentences = sent_tokenize(text) # tokenize sentences
    paraphrases = query(api_key, {"inputs": sentences})
    paraphrased_text = [response['generated_text'] for response in paraphrases]
    paraphrased_text = " ".join(paraphrased_text)
    return paraphrased_text

if __name__ == '__main__':
    parser = configparser.ConfigParser()
    parser.read("config.conf")
    api_key = parser.get("huggingface_api", "api_key")

    text = """
        The ultimate test of your knowledge is your capacity to convey it to another. 
        Best way: learning by teaching & practicing. Heard D. Mermin once describe a
        person who has a good understanding of physics but can't convey it, as being 
        a natural phenomenon, not a physicist. 
        """
    paraphrased_text = get_paraphrases(api_key, text)
    print(paraphrased_text)