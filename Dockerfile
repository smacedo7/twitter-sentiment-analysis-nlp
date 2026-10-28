FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY src ./src
COPY streamlit_app.py .
COPY models ./models

RUN useradd --create-home appuser
USER appuser

# Render and other platforms provide $PORT; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# REST API by default. For the web interface run:
#   docker run -p 8501:8501 <image> streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT}"]
