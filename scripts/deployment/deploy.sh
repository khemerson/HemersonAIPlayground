#!/bin/bash
# scripts/deployment/deploy.sh
# Script de déploiement compatible Ansible

set -euo pipefail

# Variables
PROJECT_NAME="hemerson-playgroundv1"
DOCKER_COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

# Fonctions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a deployment.log
}

check_requirements() {
    log "🔍 Vérification des prérequis..."
    
    command -v docker >/dev/null 2>&1 || { log "❌ Docker non installé"; exit 1; }
    command -v docker-compose >/dev/null 2>&1 || { log "❌ Docker Compose non installé"; exit 1; }
    
    log "✅ Prérequis OK"
}

create_env_if_missing() {
    if [ ! -f "$ENV_FILE" ]; then
        log "📝 Création du fichier .env..."
        cat > "$ENV_FILE" << EOF
# === CONFIGURATION HEMERSON PLAYGROUNDV1 ===
PROJECT_NAME=hemerson-playgroundv1
ENVIRONMENT=production

# === REDIS ===
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=

# === APPLICATION ===
HOST=0.0.0.0
PORT=8000
DEBUG=false

# === IA ENDPOINTS ===
MISTRAL_ENDPOINT=http://192.168.1.211:11434
MAGISTRAL_ENDPOINT=http://192.168.1.214:11434
PIXTRAL_ENDPOINT=http://192.168.1.212:11434
WHISPER_ENDPOINT=http://192.168.1.212:8000
TTS_ENDPOINT=http://192.168.1.214:8001
COMFYUI_ENDPOINT=http://192.168.1.213:8188

# === SÉCURITÉ ===
CHAINLIT_AUTH_SECRET=$(openssl rand -hex 32)
MAX_REQUESTS_PER_MINUTE=20
MAX_REQUESTS_PER_HOUR=100

# === LOGGING ===
LOG_LEVEL=INFO
EOF
        log "✅ Fichier .env créé"
    else
        log "✅ Fichier .env existant"
    fi
}

prepare_directories() {
    log "📁 Préparation des dossiers..."
    
    mkdir -p {data/{uploads,cache,logs},logs,docker/redis}
    chmod 755 data/{uploads,cache,logs}
    chown -R $USER:$USER data logs
    
    log "✅ Dossiers préparés"
}

build_and_start() {
    log "🏗️ Construction et démarrage des services..."
    
    # Build des images
    docker-compose build --no-cache
    
    # Démarrage des services
    docker-compose up -d
    
    log "✅ Services démarrés"
}

wait_for_services() {
    log "⏳ Attente du démarrage des services..."
    
    # Attendre Redis
    timeout 60 bash -c 'until docker-compose exec redis redis-cli ping; do sleep 2; done'
    log "✅ Redis prêt"
    
    # Attendre l'application
    timeout 120 bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    log "✅ Application prête"
    
    log "🎉 Tous les services sont opérationnels"
}

run_tests() {
    log "🧪 Tests de validation..."
    
    # Test Redis
    if docker-compose exec redis redis-cli ping | grep -q PONG; then
        log "✅ Redis: OK"
    else
        log "❌ Redis: ERREUR"
        return 1
    fi
    
    # Test Application
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log "✅ Application: OK"
    else
        log "❌ Application: ERREUR"
        return 1
    fi
    
    # Test Nginx
    if curl -f http://localhost/nginx-health >/dev/null 2>&1; then
        log "✅ Nginx: OK"
    else
        log "❌ Nginx: ERREUR"
        return 1
    fi
    
    log "✅ Tous les tests passent"
}

show_status() {
    log "📊 État des services:"
    docker-compose ps
    
    log "🌐 Accès:"
    log "  - Application: http://localhost"
    log "  - API directe: http://localhost:8000"
    log "  - Monitoring: http://localhost:9090 (si activé)"
    
    log "📋 Commandes utiles:"
    log "  - Logs: docker-compose logs -f"
    log "  - Arrêt: docker-compose down"
    log "  - Redémarrage: docker-compose restart"
}

# === DÉPLOIEMENT PRINCIPAL ===
main() {
    log "🚀 Début du déploiement HemersonAIBuild PlaygroundV1"
    
    check_requirements
    create_env_if_missing
    prepare_directories
    build_and_start
    wait_for_services
    run_tests
    show_status
    
    log "🎉 Déploiement terminé avec succès!"
}

# Exécution
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
