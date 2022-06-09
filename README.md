# Essay Companion
An NLP powered Google Chrome extension to summarize, paraphrase, get named entities, and find keyword synonyms from highlighted text.
![EssayCompanionGif](https://github.com/jackmleitch/EssayCompanion/blob/main/extension/assets/icons/Essay%20Companion%20Demo%20(short).gif)

## Model Optimization
The T5 paraphrasing model, the Bart summarization model, and the DistilBERT NER model were all converted to ONNX and then were quantized to QInt8 for faster inference. All models were then deployed to an inference endpoint using FastAPI. 

The converted models along with their respective tokenizers can be downloaded from [here](https://drive.google.com/drive/folders/1_5FM97b717669T24vRv4fmU8W1Cwr8uO?usp=sharing). The 'models' directory needs to be saved at the root of the repository. 

<p float="left">
  <img src="https://github.com/jackmleitch/EssayCompanion/blob/main/extension/assets/icons/model_latency.png" width="500" /> 
  <img src="https://github.com/jackmleitch/EssayCompanion/blob/main/extension/assets/icons/model_size.png" width="500" />
</p>

## Try It Out For Yourself
First, you can use Docker to run the model API: 
```
docker pull jackmleitch/essay-companion
docker run -p 8000:8000 -d jackmleitch/essay-companion
```
Once the API is up and running, you can clone this repository and upload the extension directory to the 'Load unpacked' section in the Extensions tab of Google Chromes settings. Then you should see the extension and be able to use it! 

I do plan on hosting the API on Azure/AWS and releasing the extension to the public at somepoint and will update this repository accordingly once I get around to it. 
