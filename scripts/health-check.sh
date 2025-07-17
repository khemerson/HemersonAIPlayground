#!/bin/bash
# scripts/health-check.sh
# Script de vérification de santé

echo "🩺 Vérification de santé HemersonAIBuild"

# Services à tester
services=(
    "http://localhost:8000:Interface Chainlit"
    "http://localhost:5555:Monitoring Flower"
    "http://ollamartx40900:11434:Mistral"
    "http://hybridworker30901:11434:Magistral"
    "http://hybridworker30903:11434:Pixtral"
    "http://hybridworker30901:8000:Whisper"
    "http://hybridworker30901:8001:TTS"
    "http://comfyuirtx40902:8188:ComfyUI"
)

echo "Service | Status | Response Time"
echo "--------|--------|---------------"

for service in "${services[@]}"; do
    IFS=':' read -r url name <<< "$service"
    
    if response_time=$(curl -o /dev/null -s -w "%{time_total}" --max-time 10 "$url"); then
        echo "$name | ✅ Online | ${response_time}s"
    else
        echo "$name | ❌ Offline | -"
    fi
done

echo ""
echo "✅ Vérification terminée"
