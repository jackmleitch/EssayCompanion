FROM python:3.8-slim

RUN apt-get update -y && \
    apt-get dist-upgrade -y && \
    apt-get install -y && \
    apt-get install build-essential -y \
    supervisor wget unzip

COPY ./src /api/api
COPY ./models /api/models
COPY requirements.txt /requirements.txt

RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]

CMD ["api.main:app", "--host", "0.0.0.0"]