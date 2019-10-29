#!/usr/bin/env bash
docker-compose down
docker container prune -f --filter='label=luigi_task_id'
docker rmi $(sudo docker images | grep code | awk '{print $3}')
VERSION=0.1
docker build -t code-challenge/download-data:$VERSION download_data
docker build -t code-challenge/make-dataset:$VERSION make_dataset
docker build -t code-challenge/model:$VERSION model
docker-compose up orchestrator
