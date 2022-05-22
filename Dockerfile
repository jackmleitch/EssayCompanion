FROM python:3.8-slim

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y && \
    apt-get install build-essential -y \
    supervisor wget unzip

WORKDIR /Users/Jack/Documents/projects/EssayCompanion/

COPY onnx_T5_model.py /Users/Jack/Documents/projects/EssayCompanion/src/paraphrasing_model/onnx_T5_model.py
COPY main.py /Users/Jack/Documents/projects/EssayCompanion/src/main.py
COPY requirements.txt /Users/Jack/Documents/projects/EssayCompanion/requirements.txt

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]