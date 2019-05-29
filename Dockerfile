FROM python:alpine3.7

WORKDIR /usr/src/app

RUN pip install -r python/requirements.txt

