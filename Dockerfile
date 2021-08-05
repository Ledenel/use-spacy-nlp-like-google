FROM python:3.8.7

RUN pip install --no-cache-dir spacy==3.1.1

RUN python -m spacy download zh_core_web_sm

