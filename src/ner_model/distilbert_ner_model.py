from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from spacy.displacy import render
from typing import List

class NERModel:
    '''
    Load name entity recognition model and build prediction pipeline
    '''
    def __init__(self, 
        model_ckpt="elastic/distilbert-base-uncased-finetuned-conll03-english") -> None:
        self.model_ckpt = model_ckpt
        self.torch_device = "cpu"
        print(f"Loading {model_ckpt} tokenizer and model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForTokenClassification.from_pretrained(model_ckpt)
        self.id2label = self.model.config.id2label
        print("Model loaded!")
    
    def ner_pipeline(self, query: str) -> List[dict]:
        '''
        Build ner pipeline and output parsed model output
        :param query: query to pass to the model
        '''
        pipe = pipeline(model=self.model_ckpt, tokenizer=self.tokenizer, task='ner',
            aggregation_strategy="simple")
        return pipe(query)

    def render_ner_results_html(self, query: str) -> None:
        """
        Visualize NER results using SpaCy render
        :param query: query to pass to the model
        """
        model_outputs = self.ner_pipeline(query)
        entities = []
        for model_output in model_outputs:
            entry = {}
            entry['start'] = model_output['start']
            entry['end'] = model_output['end']
            entry['label'] = model_output['entity_group']
            entities.append(entry)
        render_data = [{'text': query, 'ents': entities, 'title': None}]
        render(render_data, style="ent", manual=True, jupyter=True)

if __name__ == "__main__":
    query = "Jack Sparrow lives in New York!"
    model = NERModel()
    results = model.ner_pipeline(query)
    print(results)



