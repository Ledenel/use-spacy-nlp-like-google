FROM python:3.8.7

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download zh_core_web_sm

