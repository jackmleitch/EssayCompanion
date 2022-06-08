# Essay Companion
An NLP powered Google Chrome extension to summarize, paraphrase, get named entities, and find keyword synonyms from highlighted text.
![EssayCompanionGif](https://github.com/jackmleitch/EssayCompanion/blob/main/extension/assets/icons/Essay%20Companion%20Demo%20(short).gif)

The T5 paraphrasing model, the Bart summarization model, and the DistilBERT NER model were all converted to ONNX and then were quantized to QInt8 for faster inference. All models were then deployed to an inference endpoint using FastAPI. 

The converted models along with their respective tokenizers can be downloaeded from [here](https://drive.google.com/drive/folders/1_5FM97b717669T24vRv4fmU8W1Cwr8uO?usp=sharing). They need to be stored in a directory called 'models' at the root of the repository. 
