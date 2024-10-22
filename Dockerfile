FROM python:3.11-slim


WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py audio.py ./

# Especifica el comando por defecto para ejecutar el archivo Python
CMD ["python", "main.py"]
