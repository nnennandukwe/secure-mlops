FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
RUN python train.py

COPY app.py .
COPY model.pkl .

CMD ["python", "app.py"]