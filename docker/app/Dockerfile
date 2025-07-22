# docker/app/Dockerfile
# Multi-stage build pour optimisation

# === STAGE 1: BUILD ===
FROM python:3.10-slim as builder

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Installer les dépendances système pour le build
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libsndfile1-dev \
    libmagic1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Créer et activer l'environnement virtuel
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copier et installer les requirements
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt

# === STAGE 2: PRODUCTION ===
FROM python:3.10-slim as production

# Variables d'environnement
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Installer seulement les dépendances runtime
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    libmagic1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copier l'environnement virtuel depuis le stage build
COPY --from=builder /opt/venv /opt/venv

# Créer l'utilisateur non-root pour sécurité
RUN groupadd -r hemerson && useradd -r -g hemerson hemerson

# Créer les dossiers avec permissions
RUN mkdir -p /app /data/uploads /data/cache /data/logs && \
    chown -R hemerson:hemerson /app /data

# Changer vers l'utilisateur non-root
USER hemerson

# Définir le répertoire de travail
WORKDIR /app

# Copier le code de l'application
COPY --chown=hemerson:hemerson app/ ./

# Exposer le port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande de démarrage
CMD ["chainlit", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]
