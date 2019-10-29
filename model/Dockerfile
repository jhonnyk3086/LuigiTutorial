FROM continuumio/anaconda3:latest

COPY . /opt/model

RUN pip install -r /opt/model/requirements.txt

USER root

WORKDIR /opt/model
