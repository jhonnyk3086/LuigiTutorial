FROM continuumio/anaconda3:latest
ENV PYTHONPATH=/opt/orchestrator:$PYTHONPATH

# copy and insall requirements only
COPY requirements.txt /opt/orchestrator/requirements.txt
RUN pip install -r /opt/orchestrator/requirements.txt

# copy all files
COPY . /opt/orchestrator/

WORKDIR /opt/orchestrator/
