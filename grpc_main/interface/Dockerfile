FROM python:3.9

RUN apt-get update \
  && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential

WORKDIR /home/src

COPY interface/requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY interface/src .
COPY protos/backend.proto .

RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. ./backend.proto
