# app/app.py
"""
Application Chainlit principale pour HemersonAIBuild
Interface conversationnelle optimisée pour votre infrastructure IA
"""

import chainlit as cl
import asyncio
import time
from typing import Dict, Any, Optional, List
from loguru import logger
import base64
import io
from PIL import Image

# Imports des services
from services.ai_services import ai_services
from utils.rate_limiter import rate_limiter
from utils.config import settings, AI_MODELS, AUDIO_SERVICES

# Configuration du logging
logger.add("logs/chainlit_app.log", rotation="1 MB", retention="7 days", level="INFO")

# === ÉVÉNEMENTS DE CYCLE DE VIE ===

@cl.on_chat_start
async def on_chat_start():
    """
    Événement déclenché au démarrage d'une nouvelle session de chat
    
    Initialise la session utilisateur et affiche le message d'accueil
    """
    try:
        # Générer un ID utilisateur unique
        user_id = cl.user_session.get("user_id", f"user_{int(time.time())}")
        cl.user_session.set("user_id", user_id)
        
        # Initialiser les données de session
        cl.user_session.set("request_count", 0)
        cl.user_session.set("session_start", time.time())
        cl.user_session.set("conversation_history", [])
        cl.user_session.set("uploaded_files", [])
        
        logger.info(f"🚀 Nouvelle session démarrée pour {user_id}")
        
        # Vérifier l'état des services
        health_status = await ai_services.health_check()
        online_services = health_status["summary"]["online"]
        total_services = health_status["summary"]["total"]
        
        # Message d'accueil personnalisé
        welcome_message = f"""
        {settings.WELCOME_MESSAGE}
        
        **🔧 État des Services :**
        - **Services actifs** : {online_services}/{total_services}
        - **Santé du système** : {health_status["summary"]["health_percentage"]:.0f}%
        
        **Pour commencer, vous pouvez :**
        1. **Poser une question** pour tester Mistral
        2. **Uploader une image** pour l'analyse avec Pixtral
        3. **Envoyer un audio** pour la transcription
        4. **Utiliser les actions** ci-dessous
        """
        
        # Actions disponibles
        actions = [
            cl.Action(
                name="cogito",
                label="🧠 Cogito",
                description="Analyse approfondie avec Magistral",
                icon="🧠"
            ),
            cl.Action(
                name="health_check",
                label="🔧 État Services",
                description="Vérifier l'état de l'infrastructure",
                icon="🔧"
            ),
            cl.Action(
                name="demo_audio",
                label="🎵 Démo Audio",
                description="Test du pipeline audio complet",
                icon="🎵"
            ),
            cl.Action(
                name="demo_image",
                label="🎨 Démo Image",
                description="Générer une image avec ComfyUI",
                icon="🎨"
            )
        ]
        
        # Envoyer le message d'accueil avec les actions
        await cl.Message(
            content=welcome_message,
            actions=actions
        ).send()
        
        # Afficher les métriques d'usage
        await show_usage_metrics(user_id)
        
    except Exception as e:
        logger.error(f"Erreur démarrage session: {e}")
        await cl.Message(
            content="❌ Erreur lors de l'initialisation. Veuillez rafraîchir la page."
        ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """
    Événement déclenché à chaque message de l'utilisateur
    
    Traite le message selon son type et les fichiers joints
    """
    try:
        user_id = cl.user_session.get("user_id")
        
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "chat")
        
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        # Incrémenter le compteur de requêtes
        request_count = cl.user_session.get("request_count", 0) + 1
        cl.user_session.set("request_count", request_count)
        
        # Démarrer l'indicateur de traitement
        async with cl.Step(name="processing", type="llm") as step:
            
            # Déterminer le type de traitement selon les fichiers joints
            if message.elements:
                await handle_file_upload(message)
            elif len(message.content) > 0:
                await handle_text_message(message.content, user_id)
            else:
                await cl.Message(
                    content="❓ Veuillez envoyer un message ou un fichier à analyser."
                ).send()
    
    except Exception as e:
        logger.error(f"Erreur traitement message: {e}")
        await cl.Message(
            content="❌ Erreur lors du traitement. Veuillez réessayer."
        ).send()

@cl.action_callback("cogito")
async def on_cogito_action(action):
    """
    Action Cogito : Analyse approfondie avec Magistral
    """
    try:
        user_id = cl.user_session.get("user_id")
        
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "cogito")
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        # Demander le sujet à analyser
        res = await cl.AskUserMessage(
            content="🧠 **Analyse Cogito**\n\nQuel sujet souhaitez-vous analyser en profondeur ?",
            timeout=30
        ).send()
        
        if res:
            async with cl.Step(name="cogito_analysis", type="llm") as step:
                step.input = res.content
                
                # Appeler Magistral pour l'analyse
                result = await ai_services.chat_magistral(res.content)
                
                if result['success']:
                    step.output = result['response']
                    
                    # Envoyer la réponse avec métadonnées
                    await cl.Message(
                        content=f"🧠 **Analyse Cogito - Magistral**\n\n{result['response']}\n\n---\n⚡ *Traité en {result['processing_time']:.2f}s sur {result['gpu']}*"
                    ).send()
                else:
                    await cl.Message(
                        content=f"❌ **Erreur Cogito** : {result['error']}"
                    ).send()
                    
                # Enregistrer l'usage
                await rate_limiter.record_usage(user_id, "cogito", result['success'])
    
    except Exception as e:
        logger.error(f"Erreur action Cogito: {e}")
        await cl.Message(content="❌ Erreur lors de l'analyse Cogito").send()

@cl.action_callback("health_check")
async def on_health_check_action(action):
    """
    Action Health Check : Vérifier l'état des services
    """
    try:
        async with cl.Step(name="health_check", type="tool") as step:
            step.input = "Vérification de l'état des services..."
            
            # Vérifier l'état de tous les services
            health_status = await ai_services.health_check()
            
            # Construire le rapport de santé
            report = "🔧 **État de l'Infrastructure HemersonAIBuild**\n\n"
            
            for service, status in health_status["services"].items():
                status_icon = "🟢" if status["status"] == "online" else "🔴"
                report += f"{status_icon} **{service.upper()}** : {status['status']}\n"
                if status["status"] == "online":
                    report += f"   ⚡ Temps de réponse : {status.get('response_time', 0):.3f}s\n"
                else:
                    report += f"   ❌ Erreur : {status.get('error', 'Service indisponible')}\n"
                report += "\n"
            
            # Résumé global
            summary = health_status["summary"]
            report += f"**📊 Résumé Global :**\n"
            report += f"- Services actifs : {summary['online']}/{summary['total']}\n"
            report += f"- Santé du système : {summary['health_percentage']:.0f}%\n"
            report += f"- Dernière vérification : {time.strftime('%H:%M:%S')}\n"
            
            step.output = report
            await cl.Message(content=report).send()
    
    except Exception as e:
        logger.error(f"Erreur health check: {e}")
        await cl.Message(content="❌ Erreur lors de la vérification").send()

@cl.action_callback("demo_audio")
async def on_demo_audio_action(action):
    """
    Action Démo Audio : Test du pipeline audio complet
    """
    try:
        # Demander un fichier audio
        audio_file = await cl.AskFileMessage(
            content="🎵 **Démo Pipeline Audio**\n\nUploadez un fichier audio pour tester le pipeline complet :\n\n📤 **Audio** → 🎙️ **Whisper** → 💬 **Mistral** → 🔊 **TTS**",
            accept=["audio/*"],
            max_size_mb=10,
            timeout=60
        ).send()
        
        if audio_file:
            async with cl.Step(name="audio_pipeline", type="llm") as step:
                step.input = f"Traitement audio: {audio_file.name}"
                
                # Lire le fichier audio
                audio_data = audio_file.content
                
                # Exécuter le pipeline complet
                result = await ai_services.process_audio_pipeline(audio_data)
                
                if result['success']:
                    # Afficher les résultats du pipeline
                    pipeline_report = f"""
🎵 **Pipeline Audio Complet**

🎙️ **Transcription (Whisper)** :
> {result['transcription']}

💬 **Réponse (Mistral)** :
{result['response']}

⚡ **Performance** :
- Transcription : {result['performance_breakdown']['transcription_time']:.2f}s
- Génération : {result['performance_breakdown']['generation_time']:.2f}s
- Synthèse : {result['performance_breakdown']['synthesis_time']:.2f}s
- **Total** : {result['total_time']:.2f}s
"""
                    
                    step.output = pipeline_report
                    
                    # Envoyer la réponse écrite
                    await cl.Message(content=pipeline_report).send()
                    
                    # Envoyer l'audio généré
                    if result['audio_response']:
                        audio_element = cl.Audio(
                            content=result['audio_response'],
                            name="reponse_audio.wav",
                            display="inline"
                        )
                        await cl.Message(
                            content="🔊 **Réponse Audio Générée**",
                            elements=[audio_element]
                        ).send()
                else:
                    await cl.Message(
                        content=f"❌ **Erreur Pipeline** : {result['error']}"
                    ).send()
    
    except Exception as e:
        logger.error(f"Erreur démo audio: {e}")
        await cl.Message(content="❌ Erreur lors de la démo audio").send()

@cl.action_callback("demo_image")
async def on_demo_image_action(action):
    """
    Action Démo Image : Génération d'image avec ComfyUI
    """
    try:
        user_id = cl.user_session.get("user_id")
        
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "image_generation")
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        # Demander le prompt
        res = await cl.AskUserMessage(
            content="🎨 **Génération d'Image**\n\nDécrivez l'image que vous souhaitez générer :",
            timeout=30
        ).send()
        
        if res:
            async with cl.Step(name="image_generation", type="tool") as step:
                step.input = res.content
                
                # Générer l'image
                result = await ai_services.generate_image_comfyui(res.content)
                
                if result['success']:
                    step.output = f"Image générée en {result['processing_time']:.2f}s"
                    
                    # Créer l'élément image
                    if result['image_data']:
                        image_element = cl.Image(
                            content=base64.b64decode(result['image_data']),
                            name="image_generee.png",
                            display="inline"
                        )
                        
                        await cl.Message(
                            content=f"🎨 **Image Générée**\n\n**Prompt** : {res.content}\n\n⚡ *Généré en {result['processing_time']:.2f}s sur {result['gpu']}*",
                            elements=[image_element]
                        ).send()
                else:
                    await cl.Message(
                        content=f"❌ **Erreur Génération** : {result['error']}"
                    ).send()
                    
                # Enregistrer l'usage
                await rate_limiter.record_usage(user_id, "image_generation", result['success'])
    
    except Exception as e:
        logger.error(f"Erreur démo image: {e}")
        await cl.Message(content="❌ Erreur lors de la génération d'image").send()

# === FONCTIONS UTILITAIRES ===

async def handle_text_message(text: str, user_id: str):
    """
    Traite un message texte standard
    """
    try:
        async with cl.Step(name="text_processing", type="llm") as step:
            step.input = text
            
            # Traiter avec Mistral
            result = await ai_services.chat_mistral(text)
            
            if result['success']:
                step.output = result['response']
                
                # Envoyer la réponse
                await cl.Message(
                    content=f"{result['response']}\n\n---\n⚡ *{result['model']} sur {result['gpu']} - {result['processing_time']:.2f}s*"
                ).send()
            else:
                await cl.Message(
                    content=f"❌ **Erreur** : {result['error']}"
                ).send()
                
            # Enregistrer l'usage
            await rate_limiter.record_usage(user_id, "chat", result['success'])
    
    except Exception as e:
        logger.error(f"Erreur traitement texte: {e}")
        await cl.Message(content="❌ Erreur lors du traitement").send()

async def handle_file_upload(message: cl.Message):
    """
    Traite les fichiers uploadés
    """
    try:
        user_id = cl.user_session.get("user_id")
        
        for element in message.elements:
            if isinstance(element, cl.Image):
                await handle_image_analysis(element, user_id, message.content)
            elif isinstance(element, cl.Audio):
                await handle_audio_transcription(element, user_id)
            elif isinstance(element, cl.File):
                await handle_document_rag(element, user_id, message.content)
    
    except Exception as e:
        logger.error(f"Erreur traitement fichier: {e}")
        await cl.Message(content="❌ Erreur lors du traitement du fichier").send()

async def handle_image_analysis(image_element: cl.Image, user_id: str, prompt: str = ""):
    """
    Analyse une image avec Pixtral
    """
    try:
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "image_analysis")
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        async with cl.Step(name="image_analysis", type="llm") as step:
            step.input = f"Analyse d'image: {image_element.name}"
            
            # Prompt par défaut ou personnalisé
            analysis_prompt = prompt if prompt else "Décris cette image en détail"
            
            # Analyser avec Pixtral
            result = await ai_services.analyze_image_pixtral(
                image_element.content, 
                analysis_prompt
            )
            
            if result['success']:
                step.output = result['response']
                
                # Envoyer l'analyse
                await cl.Message(
                    content=f"🖼️ **Analyse d'Image - Pixtral**\n\n{result['response']}\n\n---\n⚡ *Analysé en {result['processing_time']:.2f}s sur {result['gpu']}*"
                ).send()
            else:
                await cl.Message(
                    content=f"❌ **Erreur Analyse** : {result['error']}"
                ).send()
                
            # Enregistrer l'usage
            await rate_limiter.record_usage(user_id, "image_analysis", result['success'])
    
    except Exception as e:
        logger.error(f"Erreur analyse image: {e}")
        await cl.Message(content="❌ Erreur lors de l'analyse d'image").send()

async def handle_audio_transcription(audio_element: cl.Audio, user_id: str):
    """
    Transcrit un fichier audio avec Whisper
    """
    try:
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "audio_transcription")
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        async with cl.Step(name="audio_transcription", type="llm") as step:
            step.input = f"Transcription audio: {audio_element.name}"
            
            # Transcrire avec Whisper
            result = await ai_services.transcribe_whisper(audio_element.content)
            
            if result['success']:
                step.output = result['transcription']
                
                # Envoyer la transcription
                await cl.Message(
                    content=f"🎙️ **Transcription - Whisper V3**\n\n> {result['transcription']}\n\n---\n⚡ *Transcrit en {result['processing_time']:.2f}s sur {result['gpu']}*"
                ).send()
                
                # Proposer de traiter la transcription
                if result['transcription'].strip():
                    await cl.Message(
                        content="💬 Souhaitez-vous que je traite cette transcription avec Mistral ?"
                    ).send()
            else:
                await cl.Message(
                    content=f"❌ **Erreur Transcription** : {result['error']}"
                ).send()
                
            # Enregistrer l'usage
            await rate_limiter.record_usage(user_id, "audio_transcription", result['success'])
    
    except Exception as e:
        logger.error(f"Erreur transcription audio: {e}")
        await cl.Message(content="❌ Erreur lors de la transcription").send()

async def handle_document_rag(file_element: cl.File, user_id: str, query: str = ""):
    """
    Traite un document pour RAG temporaire
    """
    try:
        # Vérifier les limitations
        is_allowed, limit_info = await rate_limiter.is_allowed(user_id, "document_rag")
        if not is_allowed:
            await send_rate_limit_message(limit_info)
            return
        
        async with cl.Step(name="document_processing", type="tool") as step:
            step.input = f"Traitement document: {file_element.name}"
            
            # Extraire le contenu selon le type
            content = await extract_document_content(file_element)
            
            if content:
                # Stocker temporairement dans la session
                docs = cl.user_session.get("uploaded_files", [])
                docs.append({
                    "name": file_element.name,
                    "content": content,
                    "timestamp": time.time()
                })
                cl.user_session.set("uploaded_files", docs)
                
                step.output = f"Document traité: {len(content)} caractères"
                
                # Traiter la requête si fournie
                if query:
                    # Rechercher dans le document
                    relevant_content = search_in_document(content, query)
                    
                    # Générer une réponse contextualisée
                    prompt = f"""
                    Basé sur ce document "{file_element.name}":
                    
                    {relevant_content}
                    
                    Question: {query}
                    
                    Réponds en te basant uniquement sur le contenu du document.
                    """
                    
                    result = await ai_services.chat_mistral(prompt)
                    
                    if result['success']:
                        await cl.Message(
                            content=f"📄 **Réponse basée sur {file_element.name}**\n\n{result['response']}"
                        ).send()
                else:
                    await cl.Message(
                        content=f"📄 **Document traité** : {file_element.name}\n\nVous pouvez maintenant poser des questions sur ce document."
                    ).send()
            else:
                await cl.Message(
                    content="❌ Impossible d'extraire le contenu du document"
                ).send()
    
    except Exception as e:
        logger.error(f"Erreur traitement document: {e}")
        await cl.Message(content="❌ Erreur lors du traitement du document").send()

async def extract_document_content(file_element: cl.File) -> str:
    """
    Extrait le contenu d'un document selon son type
    """
    try:
        file_extension = file_element.name.lower().split('.')[-1]
        
        if file_extension == 'txt':
            return file_element.content.decode('utf-8')
        elif file_extension == 'pdf':
            # Traitement PDF (à implémenter selon vos besoins)
            return "Contenu PDF à extraire"
        elif file_extension in ['doc', 'docx']:
            # Traitement Word (à implémenter selon vos besoins)
            return "Contenu Word à extraire"
        else:
            return ""
    
    except Exception as e:
        logger.error(f"Erreur extraction contenu: {e}")
        return ""

def search_in_document(content: str, query: str) -> str:
    """
    Recherche simple dans un document
    """
    # Implémentation basique - à améliorer selon vos besoins
    lines = content.split('\n')
    relevant_lines = []
    
    for line in lines:
        if any(word.lower() in line.lower() for word in query.split()):
            relevant_lines.append(line)
    
    return '\n'.join(relevant_lines[:10])  # Limiter à 10 lignes

async def send_rate_limit_message(limit_info: Dict):
    """
    Envoie un message d'information sur les limitations
    """
    message = f"""
🛡️ **Limite de Requêtes Atteinte**

**Niveau** : {limit_info['tier_description']}
**Type** : {limit_info['request_type']}

**Limites** :
- Par minute : {limit_info['rpm_used']}/{limit_info['rpm_limit']} (reste: {limit_info['rpm_remaining']})
- Par heure : {limit_info['rph_used']}/{limit_info['rph_limit']} (reste: {limit_info['rph_remaining']})

**Prochaine réinitialisation** : {time.strftime('%H:%M', time.localtime(limit_info['next_reset']['minute']))}

*Veuillez patienter avant de refaire une requête.*
"""
    
    await cl.Message(content=message).send()

async def show_usage_metrics(user_id: str):
    """
    Affiche les métriques d'utilisation
    """
    try:
        stats = await rate_limiter.get_user_stats(user_id)
        
        if "error" not in stats:
            metrics = f"""
📊 **Vos Métriques d'Usage**

**Niveau** : {stats['current_limits']['tier_description']}
**Session** : {stats['current_limits']['rpm_used']} requêtes utilisées

*Profitez de votre infrastructure IA souveraine !*
"""
            await cl.Message(content=metrics).send()
    
    except Exception as e:
        logger.error(f"Erreur métriques: {e}")

# === POINT D'ENTRÉE ===

if __name__ == "__main__":
    # Configuration du logging
    logger.info("🚀 Démarrage HemersonAIBuild Chainlit Playground")
    
    # Lancement de l'application
    cl.run(
        host=settings.CHAINLIT_HOST,
        port=settings.CHAINLIT_PORT,
        headless=False,
        debug=settings.DEBUG
    )
