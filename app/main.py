# app/main.py
"""Application PlaygroundV1 HemersonAIBuild"""

import chainlit as cl
from loguru import logger

@cl.on_chat_start
async def on_chat_start():
    await cl.Message(content="🚀 PlaygroundV1 Test").send()

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(content=f"Echo: {message.content}").send()

if __name__ == "__main__":
    logger.info("🚀 Démarrage PlaygroundV1")

# app/main.py - Ajouts pour l'analyse d'images

# Ajouter ces imports
from components.vision_interface import vision_interface
from services.vision.pixtral_service import pixtral_service

# Ajouter dans handle_file_uploads
async def handle_image_analysis(element: cl.File, prompt: str = None):
    """Analyse d'image avec Pixtral - Intégration complète"""
    try:
        # Convertir cl.File en cl.Image si nécessaire
        image_element = cl.Image(
            name=element.name,
            content=element.content,
            display="inline"
        )
        
        # Déterminer le type d'analyse selon le prompt
        analysis_type = "comprehensive"  # Par défaut
        
        if prompt:
            prompt_lower = prompt.lower()
            if any(keyword in prompt_lower for keyword in ['technique', 'technical', 'qualité', 'résolution']):
                analysis_type = "technical"
            elif any(keyword in prompt_lower for keyword in ['créatif', 'artistique', 'style', 'art']):
                analysis_type = "creative"
            elif any(keyword in prompt_lower for keyword in ['document', 'texte', 'lecture', 'extraction']):
                analysis_type = "document"
        
        # Lancer l'analyse
        await vision_interface.handle_image_upload(
            image_element, 
            prompt, 
            analysis_type
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur analyse image: {e}")
        await cl.Message(
            content=f"❌ **Erreur d'analyse d'image** : {str(e)}"
        ).send()

# Ajouter ces actions dans on_chat_start
actions.extend([
    cl.Action(
        name="image_analysis_demo",
        label="🖼️ Demo Analyse Image",
        description="Démonstration des capacités d'analyse d'images",
        icon="🖼️"
    ),
    cl.Action(
        name="vision_metrics",
        label="📊 Métriques Vision", 
        description="Statistiques d'analyse d'images",
        icon="📊"
    )
])

# Ajouter les handlers d'actions
@cl.action_callback("image_analysis_demo")
async def on_image_analysis_demo(action: cl.Action):
    """Démonstration des capacités d'analyse d'images"""
    try:
        demo_content = """
🖼️ **Démonstration Analyse d'Images avec Pixtral**

## 🚀 Capacités Disponibles

### 📊 **Analyse Complète** (Recommandée)
- Description détaillée de tous les éléments
- Composition, couleurs, style artistique
- Contexte et signification
- Observations techniques et créatives

### 🔧 **Analyse Technique**
- Qualité et résolution de l'image
- Technique de création (photo, rendu, dessin)
- Aspects techniques remarquables
- Suggestions d'optimisation

### 🎨 **Analyse Créative**
- Approche artistique et esthétique
- Style et influences créatives
- Potentiel d'amélioration artistique
- Inspiration et références

### 📄 **Analyse Document**
- Extraction d'informations textuelles
- Structure et mise en page
- Optimisation pour OCR
- Contenu informatif structuré

## 📤 Comment Utiliser

1. **Glissez-déposez** une image dans le chat
2. **Ajoutez un prompt** (optionnel) pour guider l'analyse
3. **Attendez l'analyse** automatique avec Pixtral 12B
4. **Explorez** les résultats détaillés avec métriques

## 🎯 Formats Supportés
- **Images** : JPEG, PNG, WEBP, GIF, BMP
- **Taille max** : 25MB par image
- **Résolution** : Optimale jusqu'à 2048x2048
- **Multi-images** : Jusqu'à 4 images simultanément

## ⚡ Performances
- **GPU** : RTX 4090 (192.168.1.212)
- **Modèle** : Pixtral 12B Multimodal
- **Cache** : Intelligent pour réponses rapides
- **Temps** : ~5-15s selon complexité

---
**💡 Conseil** : Uploadez une image pour commencer !
"""
        
        await cl.Message(content=demo_content).send()
        
    except Exception as e:
        logger.error(f"❌ Erreur demo analyse: {e}")

@cl.action_callback("vision_metrics")
async def on_vision_metrics(action: cl.Action):
    """Affiche les métriques d'analyse d'images"""
    try:
        # Récupérer les métriques
        metrics = await pixtral_service.get_vision_metrics()
        
        report = f"""
📊 **Métriques Analyse d'Images - Pixtral**

## 📈 Statistiques Globales
- **Images analysées** : {metrics['images_analyzed']}
- **Analyses réussies** : {metrics['successful_analyses']}
- **Taux de succès** : {metrics['success_rate_percent']:.1f}%
- **Cache hit rate** : {metrics['cache_hit_rate_percent']:.1f}%

## ⚡ Performance
- **Temps préparation moyen** : {metrics['preprocessing_time_avg']:.2f}s
- **Temps analyse moyen** : {metrics['analysis_time_avg']:.2f}s
- **Résolution moyenne** : {metrics['avg_image_resolution'] / 1000000:.1f} MP

## 📊 Distribution Formats
- **Format le plus utilisé** : {metrics.get('most_common_format', 'N/A')}
- **Formats traités** : {', '.join(metrics['format_distributions'].keys()) if metrics['format_distributions'] else 'Aucun'}

## 🖥️ Infrastructure
- **Modèle** : Pixtral 12B Multimodal  
- **GPU** : RTX 4090 (192.168.1.212)
- **Cache** : Redis intelligent
- **Optimisations** : Activées

---
*Métriques en temps réel depuis le démarrage*
"""
        
        await cl.Message(content=report).send()
        
    except Exception as e:
        logger.error(f"❌ Erreur métriques vision: {e}")
        await cl.Message(
            content="❌ Erreur lors de la récupération des métriques vision."
        ).send()

# app/main.py - Ajouts pour la génération d'images

# Ajouter ces imports
from components.image_generation_interface import image_gen_interface
from services.image_generation.comfyui_service import comfyui_service

# Ajouter dans on_chat_start
actions.extend([
    cl.Action(
        name="generate_image",
        label="🎨 Générer Image",
        description="Créer une image avec ComfyUI",
        icon="🎨"
    ),
    cl.Action(
        name="image_gallery",
        label="🖼️ Galerie Exemples",
        description="Voir des exemples de génération",
        icon="🖼️"
    ),
    cl.Action(
        name="comfyui_metrics",
        label="📊 Métriques ComfyUI",
        description="Statistiques de génération",
        icon="📊"
    )
])

# Ajouter les handlers d'actions
@cl.action_callback("generate_image")
async def on_generate_image(action: cl.Action):
    """Génération d'image avec ComfyUI"""
    try:
        await image_gen_interface.handle_image_generation_request()
    except Exception as e:
        logger.error(f"❌ Erreur génération image: {e}")
        await cl.Message(
            content=f"❌ **Erreur génération d'image** : {str(e)}"
        ).send()

@cl.action_callback("image_gallery")
async def on_image_gallery(action: cl.Action):
    """Galerie d'exemples de génération"""
    try:
        await image_gen_interface.show_generation_gallery()
    except Exception as e:
        logger.error(f"❌ Erreur galerie: {e}")

@cl.action_callback("comfyui_metrics")
async def on_comfyui_metrics(action: cl.Action):
    """Métriques de génération ComfyUI"""
    try:
        # Récupérer les métriques
        metrics = await comfyui_service.get_generation_metrics()
        available_models = await comfyui_service.get_available_models()
        
        report = f"""
🎨 **Métriques ComfyUI - Génération d'Images**

## 📊 Statistiques Globales
- **Images générées** : {metrics['images_generated']}
- **Générations réussies** : {metrics['successful_generations']}
- **Taux de succès** : {metrics['success_rate_percent']:.1f}%
- **Cache hit rate** : {metrics['cache_hit_rate_percent']:.1f}%

## ⚡ Performance
- **Temps moyen de génération** : {metrics['average_generation_time']:.2f}s
- **Temps total** : {metrics['total_generation_time'] / 60:.1f} minutes
- **Efficacité vs estimation** : {metrics['efficiency_metrics']['avg_time_vs_estimated']:.1f}x

## 🛠️ Utilisation
- **Workflow le plus utilisé** : {metrics['most_used_workflow'].title()}
- **Style le plus populaire** : {metrics['most_popular_style'].title()}

## 📈 Distribution Workflows
"""
        
        for workflow, count in metrics.get('workflow_usage', {}).items():
            percentage = (count / max(metrics['images_generated'], 1)) * 100
            report += f"- **{workflow.title()}** : {count} ({percentage:.1f}%)\n"
        
        report += "\n## 🎨 Styles Populaires\n"
        for style, count in metrics.get('popular_styles', {}).items():
            percentage = (count / max(metrics['images_generated'], 1)) * 100
            report += f"- **{style.title()}** : {count} ({percentage:.1f}%)\n"
        
        report += f"""
## 🖥️ Infrastructure
- **GPU** : {metrics['efficiency_metrics']['gpu_utilization']}
- **Modèles disponibles** : {len(available_models.get('models', {}).get('checkpoints', []))}
- **Workflows disponibles** : {len(available_models.get('workflows', []))}
- **Styles disponibles** : {len(available_models.get('styles', []))}

---
*Métriques en temps réel depuis le démarrage*
"""
        
        await cl.Message(content=report).send()
        
    except Exception as e:
        logger.error(f"❌ Erreur métriques ComfyUI: {e}")
        await cl.Message(
            content="❌ Erreur lors de la récupération des métriques ComfyUI."
        ).send()

# Handlers pour les actions de workflow
@cl.action_callback("generate_standard")
async def on_generate_standard(action: cl.Action):
    """Génération avec workflow standard"""
    try:
        data = action.value
        await image_gen_interface.handle_workflow_selection(
            workflow="standard",
            prompt=data["prompt"],
            style="artistic"
        )
    except Exception as e:
        logger.error(f"❌ Erreur génération standard: {e}")

@cl.action_callback("generate_quality")
async def on_generate_quality(action: cl.Action):
    """Génération avec workflow qualité"""
    try:
        data = action.value
        await image_gen_interface.handle_workflow_selection(
            workflow="quality",
            prompt=data["prompt"],
            style="artistic"
        )
    except Exception as e:
        logger.error(f"❌ Erreur génération qualité: {e}")

@cl.action_callback("generate_speed")
async def on_generate_speed(action: cl.Action):
    """Génération avec workflow rapide"""
    try:
        data = action.value
        await image_gen_interface.handle_workflow_selection(
            workflow="speed",
            prompt=data["prompt"],
            style="artistic"
        )
    except Exception as e:
        logger.error(f"❌ Erreur génération rapide: {e}")

@cl.action_callback("generate_artistic")
async def on_generate_artistic(action: cl.Action):
    """Génération avec workflow artistique"""
    try:
        data = action.value
        await image_gen_interface.handle_workflow_selection(
            workflow="artistic",
            prompt=data["prompt"],
            style="artistic"
        )
    except Exception as e:
        logger.error(f"❌ Erreur génération artistique: {e}")

@cl.action_callback("generate_portrait")
async def on_generate_portrait(action: cl.Action):
    """Génération avec workflow portrait"""
    try:
        data = action.value
        await image_gen_interface.handle_workflow_selection(
            workflow="portrait",
            prompt=data["prompt"],
            style="realistic"
        )
    except Exception as e:
        logger.error(f"❌ Erreur génération portrait: {e}")
