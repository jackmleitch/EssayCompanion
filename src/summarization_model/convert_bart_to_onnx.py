from fastBart import export_and_get_onnx_model, get_onnx_model
from transformers import AutoTokenizer
from os.path import exists

model_ckpt = 'facebook/bart-large-cnn'
output_path = "models/summarization/model"

if not exists("models/summarization/model/bart-large-cnn-encoder-quantized.onnx"):
    # export model to ONNX and quantize from 32bit to 8bit
    print(f"Exporting f{model_ckpt} to ONNX and quantizing...")
    model = export_and_get_onnx_model(model_ckpt, output_path)
    print("Model exported and saved to /models directory")
else:
    print("ONNX model already exists")

if __name__ == '__main__':
    # run already exported model
    model = get_onnx_model(model_ckpt, onnx_models_path=output_path)
    tokenizer = AutoTokenizer.from_pretrained("models/summarization/tokenizer")
    paragraph = """
        For a few years, rumors have persisted that Microsoft was exploring building some form 
        of streaming stick to offer Xbox Cloud Gaming via a more affordable dongle, similarly 
        to Chromecast and Google Stadia. The first hint was Project Hobart. More recently, a code 
        name "Keystone" appeared in an Xbox OS list, lending fire to rumors that Microsoft was 
        continuing to explore additional hardware for the Xbox lineup. 

        We can now confirm that that is indeed true, and it pertains to a modernized HDMI 
        streaming device that runs Xbox Game Pass and its cloud gaming service. Microsoft is, 
        however, taking exploring additional iterations of the product before taking it to market. 
        """
    model_inputs = tokenizer(paragraph, max_length=400, truncation=True, return_tensors='pt')
    gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
    output_sequences = model.generate(**model_inputs, **gen_kwargs)
    generated_text = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)[0]
    print(paragraph + "\n")
    print(generated_text)
