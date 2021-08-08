FROM python:3.8.7

WORKDIR /app

COPY ./requirements.txt /app/

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download --direct zh_core_web_sm-3.1.0

RUN python -m spacy download --direct zh_core_web_trf-3.1.0

COPY . /app

EXPOSE 5000

ENTRYPOINT [ "flask" ]

CMD [ "run", "--host", "0.0.0.0" ]
