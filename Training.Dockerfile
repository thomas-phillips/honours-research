FROM nvidia/cuda:12.3.1-base-ubuntu22.04

RUN apt-get update
RUN apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get install -y python3.8 python3-pip libsndfile1 libsndfile1-dev nano ffmpeg libavcodec-extra

WORKDIR /app

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Set Python 3 as default for python command.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
