# Dockerfile
# Image Docker optimisée pour Chainlit

# Utilise Python 3.11 pour performance optimale
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Installe les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    curl \
    ffmpeg \
    libsndfile1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copie et installe les dépendances Python
COPY app/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copie le code de l'application
COPY app/ .

# Crée un utilisateur non-root pour la sécurité
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app

# Crée les dossiers nécessaires
RUN mkdir -p /app/uploads /app/sessions /app/logs
RUN chown -R appuser:appuser /app/uploads /app/sessions /app/logs

# Expose le port Chainlit
EXPOSE 8000

# Utilise l'utilisateur sécurisé
USER appuser

# Commande de démarrage optimisée
CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]
