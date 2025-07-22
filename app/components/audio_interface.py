# app/components/audio_interface.py
"""
Interface Chainlit pour Pipeline Audio
Optimisée pour UX et performances
"""

import chainlit as cl
from typing import Dict, Any
import time
import base64
import io

from services.audio.audio_pipeline import AudioPipelineService

class AudioInterface:
    """Interface Chainlit pour le pipeline audio"""
    
    def __init__(self, audio_pipeline: AudioPipelineService):
        self.audio_pipeline = audio_pipeline
    
    async def handle_audio_upload(self, audio_element: cl.Audio, message_content: str = None):
        """
        Traite un fichier audio uploadé avec interface progressive
        
        Args:
            audio_element: Élément audio Chainlit
            message_content: Message associé éventuel
        """
        try:
            # Validation du fichier audio
            if not self._validate_audio_file(audio_element):
                await cl.Message(
                    content="❌ **Fichier audio invalide**\n\nFormats supportés : WAV, MP3, M4A\nTaille max : 25MB\nDurée max : 5 minutes"
                ).send()
                return
            
            # Déterminer le mode d'analyse
            analysis_mode = self._determine_analysis_mode(message_content)
            
            # Interface de progression avec étapes
            progress_message = await cl.Message(
                content=f"🎵 **Pipeline Audio Démarré**\n\n**Fichier** : {audio_element.name}\n**Mode** : {analysis_mode.title()}\n**Statut** : Initialisation..."
            ).send()
            
            # Étape 1 : Transcription
            await progress_message.update(
                content=f"🎵 **Pipeline Audio en Cours**\n\n**Fichier** : {audio_element.name}\n**Mode** : {analysis_mode.title()}\n\n1️⃣ **Transcription** : En cours avec Whisper V3...\n2️⃣ **Traitement IA** : En attente\n3️⃣ **Synthèse** : En attente"
            )
            
            # Lancer le pipeline complet
            result = await self.audio_pipeline.process_complete_pipeline(
                audio_data=audio_element.content,
                filename=audio_element.name,
                analysis_mode=analysis_mode,
                response_voice="female",  # Paramètre par défaut
                language="fr"
            )
            
            if result["success"]:
                await self._display_pipeline_success(result, progress_message)
            else:
                await self._display_pipeline_error(result, progress_message)
                
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur Pipeline Audio**\n\n{str(e)}\n\nVeuillez réessayer avec un autre fichier."
            ).send()
    
    async def _display_pipeline_success(self, result: Dict[str, Any], progress_message: cl.Message):
        """Affiche le résultat d'un pipeline réussi"""
        
        # Créer l'élément audio de réponse
        response_audio = cl.Audio(
            name="response_audio.wav",
            content=result["response_audio"],
            display="inline"
        )
        
        # Métriques de performance
        metrics = result["performance_metrics"]
        quality = result["quality_indicators"]
        
        # Message de succès détaillé
        success_content = f"""
🎉 **Pipeline Audio Terminé avec Succès !**

## 📝 Transcription
**Texte reconnu** : "{result['transcription'][:200]}{'...' if len(result['transcription']) > 200 else ''}"
**Confiance** : {metrics['transcription_confidence']:.1%}
**Durée** : {metrics.get('transcription_time', 0):.2f}s

## 🤖 Réponse IA ({result['analysis_mode'].title()})
**Réponse** : "{result['ai_response'][:200]}{'...' if len(result['ai_response']) > 200 else ''}"
**Temps de traitement** : {metrics.get('ai_processing_time', 0):.2f}s

## 🔊 Synthèse Vocale
**Voix** : {result['voice'].title()}
**Taille audio** : {metrics['response_audio_size'] / 1024:.1f} KB
**Temps** : {metrics.get('synthesis_time', 0):.2f}s

## ⚡ Performance Globale
**Temps total** : {metrics['total_time']:.2f}s
**Efficacité** : {quality['pipeline_efficiency'].title()}
**Qualité transcription** : {quality.get('transcription_quality', {}).get('quality', 'unknown').title()}

---
*Écoutez la réponse audio ci-dessous* 👇
"""
        
        # Mettre à jour le message de progression
        await progress_message.update(content=success_content, elements=[response_audio])
        
        # Message séparé pour la transcription complète si longue
        if len(result['transcription']) > 200:
            await cl.Message(
                content=f"📜 **Transcription Complète**\n\n{result['transcription']}"
            ).send()
        
        # Réponse IA complète si longue
        if len(result['ai_response']) > 200:
            await cl.Message(
                content=f"🤖 **Réponse IA Complète**\n\n{result['ai_response']}"
            ).send()
    
    async def _display_pipeline_error(self, result: Dict[str, Any], progress_message: cl.Message):
        """Affiche les erreurs du pipeline avec diagnostic"""
        
        error_content = f"""
❌ **Erreur Pipeline Audio**

**Étape échouée** : {result.get('step_failed', 'Inconnue')}
**Erreur** : {result.get('error', 'Erreur inconnue')}
**Temps écoulé** : {result.get('total_time', 0):.2f}s

## 🔍 Diagnostic
"""
        
        # Ajouter des informations selon l'étape qui a échoué
        if result.get('step_failed') == 'transcription':
            error_content += """
La transcription audio a échoué. Causes possibles :
- Format audio non supporté
- Fichier audio corrompu  
- Service Whisper temporairement indisponible
- Fichier trop long (> 5 minutes)

**🔧 Solutions** :
- Vérifiez le format (WAV recommandé)
- Réduisez la durée du fichier
- Réessayez dans quelques instants
"""
        elif result.get('step_failed') == 'ai_processing':
            error_content += f"""
Le traitement IA a échoué après transcription réussie.

**Transcription obtenue** : "{result.get('transcription', 'Non disponible')[:200]}..."

**🔧 Solutions** :
- Le service IA est peut-être surchargé
- Réessayez avec un message plus court
"""
        elif result.get('step_failed') == 'speech_synthesis':
            error_content += f"""
La synthèse vocale a échoué.

**Transcription** : "{result.get('transcription', 'Non disponible')[:100]}..."
**Réponse IA** : "{result.get('ai_response', 'Non disponible')[:100]}..."

**🔧 Solutions** :
- Service TTS temporairement indisponible
- Texte trop long pour la synthèse
- Réessayez dans quelques instants
"""
        
        await progress_message.update(content=error_content)
    
    def _validate_audio_file(self, audio_element: cl.Audio) -> bool:
        """Valide un fichier audio"""
        try:
            # Vérifier la taille (25MB max)
            if len(audio_element.content) > 25 * 1024 * 1024:
                return False
            
            # Vérifier les formats supportés
            supported_formats = ['.wav', '.mp3', '.m4a', '.ogg']
            file_extension = audio_element.name.lower().split('.')[-1]
            
            if f'.{file_extension}' not in supported_formats:
                return False
            
            # Validation basique du contenu
            if len(audio_element.content) < 1024:  # Trop petit
                return False
            
            return True
            
        except Exception:
            return False
    
    def _determine_analysis_mode(self, message_content: str) -> str:
        """Détermine le mode d'analyse basé sur le message"""
        if not message_content:
            return "conversational"
        
        message_lower = message_content.lower()
        
        if any(keyword in message_lower for keyword in ['cogito', 'analyse', 'réfléchis', 'approfondis']):
            return "cogito"
        elif any(keyword in message_lower for keyword in ['expert', 'technique', 'professionnel', 'spécialisé']):
            return "expert"
        else:
            return "conversational"
