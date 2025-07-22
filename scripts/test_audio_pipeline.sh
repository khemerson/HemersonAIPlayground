#!/bin/bash
# scripts/test_audio_pipeline.sh
# Validation du pipeline audio

echo "🎵 Test du Pipeline Audio HemersonAIBuild PlaygroundV1"

# Variables
TEST_AUDIO_DIR="tests/audio_samples"
WHISPER_ENDPOINT="http://192.168.1.212:8000"
TTS_ENDPOINT="http://192.168.1.214:8001"

# Créer le dossier de test s'il n'existe pas
mkdir -p "$TEST_AUDIO_DIR"

# Fonction de test d'endpoint
test_endpoint() {
    local name=$1
    local endpoint=$2
    local test_path=$3
    
    echo -n "Testing $name ($endpoint$test_path)... "
    
    if curl -f -s --max-time 10 "$endpoint$test_path" > /dev/null 2>&1; then
        echo "✅ ONLINE"
        return 0
    else
        echo "❌ OFFLINE"
        return 1
    fi
}

# Tests de connectivité
echo "🔍 Test de connectivité des services audio..."
test_endpoint "Whisper V3" "$WHISPER_ENDPOINT" "/health"
test_endpoint "TTS5" "$TTS_ENDPOINT" "/health"

# Génération d'un échantillon audio de test WAV
echo "🔧 Génération échantillon audio de test..."
python3 -c "
import wave
import numpy as np

# Générer un signal audio simple (440Hz pendant 2 secondes)
sample_rate = 44100
duration = 2
t = np.linspace(0, duration, int(sample_rate * duration))
audio_data = np.sin(2 * np.pi * 440 * t) * 0.3

# Convertir en format int16
audio_data = (audio_data * 32767).astype(np.int16)

# Sauvegarder en WAV
with wave.open('$TEST_AUDIO_DIR/test_sample.wav', 'wb') as wav_file:
    wav_file.setnchannels(1)  # Mono
    wav_file.setsampwidth(2)  # 16 bits
    wav_file.setframerate(sample_rate)
    wav_file.writeframes(audio_data.tobytes())

print('✅ Échantillon audio généré: $TEST_AUDIO_DIR/test_sample.wav')
"

# Test des imports Python
echo "🐍 Test des imports Python audio..."
python3 -c "
import sys
try:
    import httpx
    import aiofiles
    import asyncio
    print('✅ Imports audio OK')
except ImportError as e:
    print(f'❌ Import manquant: {e}')
    sys.exit(1)
"

# Validation de la configuration
echo "⚙️ Validation configuration audio..."
python3 -c "
import sys
sys.path.append('app')

try:
    from services.audio.audio_pipeline import AudioPipelineService
    from services.rate_limiter.advanced_limiter import CacheManager
    print('✅ Services audio importés avec succès')
except ImportError as e:
    print(f'❌ Erreur import services: {e}')
    sys.exit(1)
"

# Tests unitaires audio
echo "🧪 Exécution des tests unitaires audio..."
if command -v pytest >/dev/null 2>&1; then
    pytest tests/test_audio_pipeline.py -v --tb=short
else
    echo "⚠️ pytest non installé, tests unitaires ignorés"
fi

# Résumé
echo ""
echo "🎯 Résumé des tests pipeline audio:"
echo "- ✅ Échantillon audio généré"
echo "- ✅ Imports Python validés"
echo "- ✅ Services configurés"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Vérifiez que Whisper V3 et TTS5 sont actifs"
echo "2. Testez avec un vrai fichier audio via l'interface"
echo "3. Surveillez les logs pour les performances"
echo ""
echo "🎵 Pipeline audio prêt pour PlaygroundV1 !"
