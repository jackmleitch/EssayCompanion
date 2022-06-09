FROM ubuntu:20.04

COPY ./src /api/api
COPY ./models /api/models
COPY requirements.txt /requirements.txt

RUN apt-get update \
    && apt-get install python3-dev python3-pip -y \
    && pip3 install -r requirements.txt

RUN python3 -m nltk.downloader punkt wordnet omw-1.4

ENV PYTHONPATH=/api
WORKDIR /api

EXPOSE 8000

ENTRYPOINT ["uvicorn"]
CMD ["api.main:app", "--host", "0.0.0.0"]