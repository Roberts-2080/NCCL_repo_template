#syntax=docker/dockerfile:1

FROM nvcr.io/nvidia/pytorch:22.07-py3

RUN apt-get update && apt-get install -y \
		python3 \
		python3-pip \
		wget \
		unzip \
		tmux \ 
		vim \
		&& rm -rf /var/lib/apt/lists/*

WORKDIR /work_dir
COPY . .
RUN python3 -m pip install -r requirements.txt
RUN chmod +x *.sh

