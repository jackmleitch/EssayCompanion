import numpy as np
from onnxruntime import InferenceSession, SessionOptions, GraphOptimizationLevel
from transformers import AutoTokenizer
from typing import List, Tuple

def create_model_for_provider(model_path, provider='CPUExecutionProvider'):
    '''
    Create CPU inference session for ONNX runtime to boost performance
    :param model_path: path to *.onnx model
    :param provider: CPU/CUDA
    :return: onnx runtime session
    '''
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    session = InferenceSession(str(model_path), options, providers=[provider])
    session.disable_fallback()
    return session

class NEROnnxModel():
    """Build NER onnx model and aggregate results into data to be rendered"""
    def __init__(self, quant=True) -> None:
        if quant:
            model_path = "models/ner/model/model_quant.onnx"
        else: 
            model_path = "models/ner/model/model.onnx"
        self.model = create_model_for_provider(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained("models/ner/tokenizer/")
        self.id2label = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG",
            4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
    
    def __call__(self, sentence: str) -> str:
        # get model inputs and other required arrays
        model_inputs = self.tokenizer(sentence, return_tensors="np",
            return_special_tokens_mask=True, return_offsets_mapping=True)
        special_tokens_mask = model_inputs.pop("special_tokens_mask")[0]
        offset_mapping = model_inputs.pop("offset_mapping")[0]
        # pass to model
        logits = self.model.run(None, dict(model_inputs))[0]
        predictions = np.argmax(logits, axis=2)[0]
        input_ids = model_inputs["input_ids"][0]
        model_outputs = {"sentence": sentence, "input_ids": input_ids, "predictions": predictions, 
            "offset_mapping": offset_mapping, "special_tokens_mask": special_tokens_mask}
        # aggregate entitity information
        pre_entities = self.gather_pre_entities(**model_outputs)
        entities = self.aggregate_words(pre_entities)
        entities = self.group_entities(entities)
        results = self.build_final_output(sentence, entities)
        return results

    def gather_pre_entities(self, sentence: str, input_ids: np.ndarray, predictions: List[int],
            offset_mapping: List[Tuple[int, int]], special_tokens_mask: np.ndarray) -> List[dict]:
        """Fuse various numpy arrays into dicts with all the information needed for aggregation"""
        pre_entities = []
        for idx, pred in enumerate(predictions):
            # Filter special_tokens, they should only occur at the sentence boundaries 
            if special_tokens_mask[idx]:
                continue
            word = self.tokenizer.convert_ids_to_tokens(int(input_ids[idx]))
            start_ind, end_ind = offset_mapping[idx]
            word_ref = sentence[start_ind:end_ind]
            is_subword = len(word) != len(word_ref)
            if int(input_ids[idx]) == self.tokenizer.unk_token_id:
                word = word_ref
                is_subword = False
            pre_entity = {"word": word, "entity": self.id2label[pred], "start": start_ind,
                "end": end_ind, "index": idx, "is_subword": is_subword}
            pre_entities.append(pre_entity)
        return pre_entities

    def aggregate_word(self, entities: List[dict]) -> dict:
        """Aggregate sub-words together and make sure entities align"""
        word = self.tokenizer.convert_tokens_to_string([entity["word"] for entity in entities])
        entity = entities[0]["entity"]
        new_entity = {"word": word, "entity": entity, "start": entities[0]["start"],
            "end": entities[-1]["end"]}
        return new_entity

    def aggregate_words(self, entities: List[dict]) -> List[dict]:
        """Override tokens from a given word that disagree to force agreement on word boundaries"""
        word_entities = []
        word_group = None
        for entity in entities:
            if word_group is None:
                word_group = [entity]
            elif entity["is_subword"]:
                word_group.append(entity)
            else:
                word_entities.append(self.aggregate_word(word_group))
                word_group = [entity]
        # add last item
        word_entities.append(self.aggregate_word(word_group))
        return word_entities

    def group_sub_entities(self, entities: List[dict]) -> dict:
        """Group together the adjacent tokens with the same entity predicted"""
        # Get the first entity in the entity group
        entity = entities[0]["entity"].split("-")[-1]
        tokens = [entity["word"] for entity in entities]
        entity_group = {"entity_group": entity, "word": self.tokenizer.convert_tokens_to_string(tokens),
            "start": entities[0]["start"], "end": entities[-1]["end"]}
        return entity_group

    def get_tag(self, entity_name: str) -> Tuple[str, str]:
        if entity_name.startswith("B-"):
            bi = "B"
            tag = entity_name[2:]
        elif entity_name.startswith("I-"):
            bi = "I"
            tag = entity_name[2:]
        else:
            # if not in B-, I- format default to I- for continuation
            bi = "I"
            tag = entity_name
        return bi, tag

    def group_entities(self, entities: List[dict]) -> List[dict]:
        """Find and group together the adjacent tokens with the same entity predicted"""
        entity_groups = []
        entity_group_disagg = []
        for entity in entities:
            if not entity_group_disagg:
                entity_group_disagg.append(entity)
                continue
            # if the current entity is similar and adjacent to the previous entity, 
            # append it to the disaggregated entity group
            bi, tag = self.get_tag(entity["entity"])
            last_bi, last_tag = self.get_tag(entity_group_disagg[-1]["entity"])
            if tag == last_tag and bi != "B":
                # modify subword type to be previous_type
                entity_group_disagg.append(entity)
            else:
                # if the current entity is different from the previous entity
                # aggregate the disaggregated entity group
                entity_groups.append(self.group_sub_entities(entity_group_disagg))
                entity_group_disagg = [entity]
        if entity_group_disagg:
            # it's the last entity, add it to the entity groups
            entity_groups.append(self.group_sub_entities(entity_group_disagg))
        return entity_groups

    def build_final_output(self, sentence: str, entity_groups: List[dict]) -> List[dict]:
        entities = []
        for entity_group in entity_groups:
            if entity_group['entity_group'] == 'O':
                continue
            else:
                entry = {}
                entry['start'] = entity_group['start']
                entry['end'] = entity_group['end']
                entry['label'] = entity_group['entity_group']
                entities.append(entry)
        render_data = [{'text': sentence, 'ents': entities, 'title': None}]
        return render_data
    
if __name__ == '__main__':
    model_ckpt = "models/ner/model/model.onnx"
    sentence = "Jack Sparrow lives in New York!"
    # sentence = "Albert Einstein was born at Ulm, in Württemberg, Germany, on March 14, 1879. Six weeks later the family moved to Munich, where he later on began his schooling at the Luitpold Gymnasium. Later, they moved to Italy and Albert continued his education at Aarau, Switzerland and in 1896 he entered the Swiss Federal Polytechnic School in Zurich to be trained as a teacher in physics and mathematics."
    pipe = NEROnnxModel()
    results = pipe(sentence)
    print(results)

