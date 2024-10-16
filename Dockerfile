# Usa una imagen base de Python
FROM python:3.11-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo main.py al directorio de trabajo del contenedor
COPY main.py .

# Si necesitas instalar dependencias adicionales, usa el siguiente comando:
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# Especifica el comando por defecto para ejecutar el archivo Python
CMD ["python", "main.py"]
