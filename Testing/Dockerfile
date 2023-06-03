FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 libsm6 libxrender1 libxext6

COPY requirements.txt . 

RUN pip3 install -r requirements.txt --no-cache-dir

RUN pip3 install moviepy --no-cache-dir

COPY . .

WORKDIR /app/src

ENV PYTHONUNBUFFERED 1

ENTRYPOINT ["python3", "app.py"]

CMD ["--codido", "False"]