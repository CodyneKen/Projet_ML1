FROM python:3.12-slim

# Définit le répertoire de travail dans le conteneur
WORKDIR /app

# Copie le fichier API (api.py) dans le conteneur
COPY api.py .

# Copie le fichier requirements.txt contenant les dépendances
COPY requirements.txt .

# Met à jour pip
RUN pip3 install --upgrade pip

# Installe les dépendances listées dans requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose l'API sur le port 8080 de l'IP externe du conteneur
CMD uvicorn api:app --host 0.0.0.0 --port 8080 --reload
