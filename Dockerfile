FROM python:3.7-stretch

RUN apt-get -y install libc-dev

RUN pip install --upgrade pip

COPY . .

WORKDIR python

RUN python -m pip install -e ".[dev, parallel]"

WORKDIR /
