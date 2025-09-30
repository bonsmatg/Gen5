FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc libpq-dev curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY jarvis_chenn.py .

EXPOSE 8000

CMD ["uvicorn", "jarvis_chenn:app", "--host", "0.0.0.0", "--port", "8000"]