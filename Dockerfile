FROM python:3.8.7

RUN pip install --no-cache-dir spacy==3.1.1

RUN pip install --no-cache-dir whoosh==2.7.4

RUN pip install --no-cache-dir flask==2.0.1

RUN python -m spacy download --direct zh_core_web_sm-3.1.0

RUN python -m spacy download --direct zh_core_web_trf-3.1.0

WORKDIR /app

COPY . /app

EXPOSE 5000

ENTRYPOINT [ "flask" ]

CMD [ "run", "--host", "0.0.0.0" ]
