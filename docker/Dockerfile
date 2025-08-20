FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y python3 python3-pip git && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt


COPY app /app/app
ENV MODEL_ID=mistralai/Mistral-7B-Instruct-v0.2
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
