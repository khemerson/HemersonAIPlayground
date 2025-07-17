#!/bin/bash
# scripts/deploy.sh
# Script de déploiement optimisé

echo "🚀 Déploiement HemersonAIBuild Chainlit Playground"

# Vérifier les prérequis
echo "🔍 Vérification des prérequis..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker n'est pas installé"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose n'est pas installé"
    exit 1
fi

# Arrêter les services existants
echo "🛑 Arrêt des services existants..."
docker-compose down

# Construire les images
echo "🔨 Construction des images..."
docker-compose build --no-cache

# Démarrer les services
echo "🚀 Démarrage des services..."
docker-compose up -d

# Vérifier le déploiement
echo "🔍 Vérification du déploiement..."
sleep 10

# Test de santé
echo "🩺 Test de santé..."
curl -f http://localhost:8000/health || echo "⚠️ Service principal non disponible"

echo "✅ Déploiement terminé!"
echo "🌐 Interface disponible sur: http://localhost:8000"
echo "📊 Monitoring Flower: http://localhost:5555"
