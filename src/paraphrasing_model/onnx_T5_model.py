import torch
from onnxruntime import (GraphOptimizationLevel, InferenceSession, SessionOptions)
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoConfig
from transformers.modeling_outputs import Seq2SeqLMOutput
from fastT5.onnx_models import T5Encoder, T5DecoderInit, T5Decoder
import nltk
from nltk.tokenize import sent_tokenize

# download nltk punkt sentence tokenizer if it's not found in files
try:
    nltk_path = nltk.find("corpora/punkt")
except Exception:
    nltk.download('punkt')

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

class OnnxT5(T5ForConditionalGeneration):
    """
    Creates a T5 model using onnx sessions encode, decoder & init_decoder
    """
    def __init__(self, model_ckpt="ramsrigouthamg/t5-large-paraphraser-diverse-high-quality"):
        config = AutoConfig.from_pretrained(model_ckpt)
        super().__init__(config)
        encoder_session = create_model_for_provider("models/paraphrase/model/t5-large-paraphraser-diverse-high-quality-encoder-quantized.onnx")
        decoder_init_session = create_model_for_provider("models/paraphrase/model/t5-large-paraphraser-diverse-high-quality-init-decoder-quantized.onnx")
        decoder_session = create_model_for_provider("models/paraphrase/model/t5-large-paraphraser-diverse-high-quality-decoder-quantized.onnx")
        self.encoder = T5Encoder(encoder_session)
        self.decoder = T5Decoder(decoder_session)
        self.decoder_init = T5DecoderInit(decoder_init_session)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
        decoder_attention_mask=None, head_mask=None, decoder_head_mask=None,
        cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None,
        inputs_embeds=None, decoder_inputs_embeds=None, labels=None, use_cache=None,
        output_attentions=None, output_hidden_states=None, return_dict=None):

        if encoder_outputs is None:
            # convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:
            # runs only for the first time:
            init_onnx_outputs = self.decoder_init(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )
            logits, past_key_values = init_onnx_outputs

        else:
            onnx_outputs = self.decoder(decoder_input_ids, attention_mask, encoder_hidden_states,
                past_key_values)
            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

class OnnxPipeline:
    '''
    Model inference pipeline 
    '''
    def __init__(self, num_beams=5):
        self.model = OnnxT5()
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("models/paraphrase/tokenizer/")
        self.num_beams = num_beams

    def __call__(self, query: str) -> str:
        '''
        Splits up input query into sentences and makes calls to the optimized T5 model
        :param query: input query to pass to the model
        '''
        sentences = sent_tokenize(query)
        paraphrased_text = []
        for sentence in sentences:
            batch = (self.tokenizer(sentence, truncation=True, padding="longest",
                max_length=100, return_tensors="pt").to('cpu')) 
            with torch.no_grad():
                translated = self.model.generate(**batch, max_length=100, num_beams=self.num_beams,
                    num_return_sequences=1, temperature=1.5)
            result = self.tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
            paraphrased_text.append(result)
        paraphrased_text = [sentence[19:] for sentence in paraphrased_text]
        paraphrased_text = " ".join(paraphrased_text)
        return paraphrased_text

if __name__ == '__main__':

    print('Loading onnx models + tokenizer...')
    pipe = OnnxPipeline()
    print('Loaded!')
    query = """
        The ultimate test of your knowledge is your capacity to convey it to another. 
        The best way of learning is by teaching & practicing. D. Mermin once described a
        person who has a good understanding of physics but can't convey it, as being 
        a natural phenomenon, not a physicist. 
        """    
    print(pipe(query))