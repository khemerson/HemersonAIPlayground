# app/components/image_generation_interface.py
"""
Interface Chainlit pour génération d'images avec ComfyUI
UX optimisée et contrôles avancés
"""

import chainlit as cl
from typing import Dict, Any, Optional
import time
import asyncio
import base64
import io
from PIL import Image

from services.image_generation.comfyui_service import comfyui_service

class ImageGenerationInterface:
    """Interface Chainlit pour la génération d'images"""
    
    def __init__(self):
        self.workflows_info = {
            "standard": {
                "emoji": "⚖️",
                "description": "Équilibre parfait qualité/vitesse - Recommandé",
                "time": "~30s"
            },
            "quality": {
                "emoji": "💎", 
                "description": "Maximum de qualité - Pour vos créations importantes",
                "time": "~60s"
            },
            "speed": {
                "emoji": "⚡",
                "description": "Génération rapide - Tests et itérations",
                "time": "~15s"
            },
            "artistic": {
                "emoji": "🎨",
                "description": "Focus créativité - Pour l'art numérique",
                "time": "~45s"
            },
            "portrait": {
                "emoji": "👤",
                "description": "Optimisé portraits - Format vertical",
                "time": "~40s"
            }
        }
        
        self.styles_info = {
            "realistic": {
                "emoji": "📸",
                "description": "Photo réaliste haute qualité"
            },
            "artistic": {
                "emoji": "🎭",
                "description": "Art numérique créatif"
            },
            "fantasy": {
                "emoji": "🧙",
                "description": "Fantasy et fantastique"
            },
            "scifi": {
                "emoji": "🚀",
                "description": "Science-fiction futuriste"
            },
            "minimalist": {
                "emoji": "⚪",
                "description": "Design épuré minimaliste"
            }
        }
    
    async def handle_image_generation_request(self, prompt: str = None):
        """
        Gère une demande de génération d'image avec interface interactive
        
        Args:
            prompt: Prompt initial (optionnel)
        """
        try:
            # Si pas de prompt fourni, le demander
            if not prompt:
                prompt_response = await cl.AskUserMessage(
                    content="""🎨 **Génération d'Image - ComfyUI**

**Décrivez l'image que vous souhaitez créer** :

💡 **Conseils pour de meilleurs résultats** :
- Soyez spécifique sur les détails
- Mentionnez le style souhaité
- Précisez les couleurs, l'éclairage, la composition
- Évitez les descriptions trop longues

**Exemples** :
- "Un paysage de montagne au coucher de soleil, style photo réaliste"
- "Portrait d'une femme en armure futuriste, art numérique"
- "Chat mignon dans un jardin fleuri, style minimaliste"

**Tapez votre description** :""",
                    timeout=60
                ).send()
                
                if not prompt_response or not prompt_response.content:
                    await cl.Message(
                        content="⏰ **Timeout** - Aucun prompt reçu pour la génération."
                    ).send()
                    return
                
                prompt = prompt_response.content
            
            # Interface de sélection des paramètres
            await self._show_generation_interface(prompt)
            
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur génération d'image** : {str(e)}"
            ).send()
    
    async def _show_generation_interface(self, prompt: str):
        """Affiche l'interface de sélection des paramètres"""
        
        # Message avec les options de workflow
        workflows_text = "## 🛠️ Choisissez votre Workflow\n\n"
        for workflow_id, info in self.workflows_info.items():
            workflows_text += f"**{info['emoji']} {workflow_id.title()}** - {info['description']} ({info['time']})\n"
        
        workflows_text += "\n## 🎨 Styles Disponibles\n\n"
        for style_id, info in self.styles_info.items():
            workflows_text += f"**{info['emoji']} {style_id.title()}** - {info['description']}\n"
        
        # Actions pour les workflows
        workflow_actions = []
        for workflow_id, info in self.workflows_info.items():
            workflow_actions.append(
                cl.Action(
                    name=f"generate_{workflow_id}",
                    label=f"{info['emoji']} {workflow_id.title()}",
                    description=f"{info['description']} ({info['time']})",
                    value={"workflow": workflow_id, "prompt": prompt}
                )
            )
        
        await cl.Message(
            content=f"""🎨 **Génération d'Image - Configuration**

**Votre prompt** : "{prompt[:100]}{'...' if len(prompt) > 100 else ''}"

{workflows_text}

**👇 Sélectionnez un workflow pour commencer** :""",
            actions=workflow_actions[:3]  # Limiter à 3 actions principales
        ).send()
        
        # Actions supplémentaires
        if len(workflow_actions) > 3:
            await cl.Message(
                content="**🔧 Workflows Avancés** :",
                actions=workflow_actions[3:]
            ).send()
    
    async def handle_workflow_selection(self, workflow: str, prompt: str, style: str = "artistic"):
        """
        Traite la sélection d'un workflow et lance la génération
        
        Args:
            workflow: Workflow sélectionné
            prompt: Prompt de génération
            style: Style sélectionné
        """
        try:
            workflow_info = self.workflows_info.get(workflow, self.workflows_info["standard"])
            style_info = self.styles_info.get(style, self.styles_info["artistic"])
            
            # Message de progression avec détails
            progress_message = await cl.Message(
                content=f"""🎨 **Génération en Cours**

**🎯 Configuration** :
- **Prompt** : "{prompt[:100]}{'...' if len(prompt) > 100 else ''}"
- **Workflow** : {workflow_info['emoji']} {workflow.title()} ({workflow_info['time']})
- **Style** : {style_info['emoji']} {style.title()}
- **GPU** : RTX 4090 (192.168.1.213)

**⏳ Statut** : Initialisation du workflow ComfyUI..."""
            ).send()
            
            # Lancer la génération
            start_time = time.time()
            
            result = await comfyui_service.generate_image(
                prompt=prompt,
                workflow=workflow,
                style=style,
                use_cache=True
            )
            
            generation_time = time.time() - start_time
            
            if result["success"]:
                await self._display_successful_generation(result, progress_message, generation_time)
            else:
                await self._display_generation_error(result, progress_message, workflow, style)
                
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur lors de la génération** : {str(e)}"
            ).send()
    
    async def _display_successful_generation(self, result: Dict[str, Any], progress_message: cl.Message, actual_time: float):
        """Affiche le résultat d'une génération réussie"""
        try:
            # Créer l'élément image
            image_element = cl.Image(
                name="generated_image.png",
                content=result["image_data"],
                display="inline"
            )
            
            # Informations détaillées
            params = result["generation_params"]
            metadata = result["metadata"]
            image_info = result["image_info"]
            
            success_content = f"""
🎉 **Image Générée avec Succès !**

## 🖼️ Résultat
**Résolution** : {image_info['width']}x{image_info['height']}
**Taille** : {image_info['size_bytes'] / 1024:.1f} KB
**Format** : {image_info['format']}

## ⚙️ Configuration Utilisée
**Workflow** : {metadata['workflow_name']}
**Style** : {metadata['style_name']}
**Steps** : {params['steps']}
**CFG Scale** : {params['cfg_scale']}
**Sampler** : {params['sampler']}

## ⚡ Performance
**Temps de génération** : {metadata['generation_time']:.2f}s
**Temps estimé** : {metadata['estimated_vs_actual']['estimated']}s
**Efficacité** : {metadata['estimated_vs_actual']['efficiency']:.1f}x
**Score qualité** : {metadata['quality_score']:.2f}/1.0

## 🎯 Prompts
**Original** : "{params['prompt']}"
**Enrichi** : "{params['enhanced_prompt'][:200]}..."
**Négatif** : "{params['negative_prompt'][:100]}..."

---
💡 **Votre image est prête !** Vous pouvez la télécharger ou générer une variante.
"""
            
            await progress_message.update(
                content=success_content,
                elements=[image_element]
            )
            
            # Proposer des actions de suivi
            await self._suggest_follow_up_actions(result)
            
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur affichage résultat** : {str(e)}"
            ).send()
    
    async def _display_generation_error(self, result: Dict[str, Any], progress_message: cl.Message, workflow: str, style: str):
        """Affiche une erreur de génération avec diagnostic"""
        
        error_content = f"""
❌ **Échec de la Génération**

**Workflow** : {workflow.title()}
**Style** : {style.title()}
**Erreur** : {result.get('error', 'Erreur inconnue')}
**Temps écoulé** : {result.get('generation_time', 0):.2f}s

## 🔍 Diagnostic
La génération d'image a échoué. Causes possibles :

### 🛠️ Service ComfyUI
- Service temporairement indisponible
- Queue surchargée
- Modèle non chargé
- Paramètres incompatibles

### 📝 Prompt
- Description trop complexe
- Termes contradictoires
- Prompt trop long ou trop court

### ⚙️ Configuration
- Résolution trop élevée
- Steps insuffisants ou excessifs
- CFG Scale inadapté

## 🔧 Solutions Recommandées

1. **Réessayez** avec le workflow "Speed" (plus rapide)
2. **Simplifiez le prompt** (description plus directe)
3. **Attendez quelques minutes** (queue possiblement surchargée)
4. **Changez de style** (certains styles sont plus stables)

**💡 Le workflow "Standard" avec style "Artistic" offre la meilleure stabilité.**
"""
        
        await progress_message.update(content=error_content)
        
        # Actions de récupération
        retry_actions = [
            cl.Action(
                name="retry_speed",
                label="🚀 Réessayer (Rapide)",
                description="Workflow Speed avec paramètres optimisés",
                value={"retry": True, "workflow": "speed"}
            ),
            cl.Action(
                name="retry_standard",
                label="⚖️ Réessayer (Standard)", 
                description="Workflow Standard stable",
                value={"retry": True, "workflow": "standard"}
            )
        ]
        
        await cl.Message(
            content="🔄 **Options de Récupération** :",
            actions=retry_actions
        ).send()
    
    async def _suggest_follow_up_actions(self, result: Dict[str, Any]):
        """Suggère des actions de suivi après génération réussie"""
        
        params = result["generation_params"]
        
        follow_up_actions = [
            cl.Action(
                name="generate_variation",
                label="🎲 Générer une Variante",
                description="Même prompt avec seed différent",
                value={"variation": True, "base_prompt": params["prompt"]}
            ),
            cl.Action(
                name="enhance_quality",
                label="💎 Version Haute Qualité",
                description="Même image avec workflow Quality",
                value={"enhance": True, "prompt": params["prompt"]}
            ),
            cl.Action(
                name="try_different_style",
                label="🎨 Autre Style",
                description="Même prompt, style différent",
                value={"restyle": True, "prompt": params["prompt"]}
            )
        ]
        
        await cl.Message(
            content="✨ **Que souhaitez-vous faire ensuite ?**",
            actions=follow_up_actions
        ).send()
    
    async def handle_batch_generation(self, prompts: List[str], workflow: str = "standard", style: str = "artistic"):
        """
        Génération d'images par batch (multiple prompts)
        
        Args:
            prompts: Liste de prompts
            workflow: Workflow à utiliser
            style: Style à appliquer
        """
        try:
            if len(prompts) > 5:
                await cl.Message(
                    content="❌ **Limite dépassée** - Maximum 5 images par batch."
                ).send()
                return
            
            batch_progress = await cl.Message(
                content=f"""🎨 **Génération Batch Démarrée**

**Nombre d'images** : {len(prompts)}
**Workflow** : {workflow.title()}
**Style** : {style.title()}

**⏳ Statut** : Préparation du batch..."""
            ).send()
            
            results = []
            
            for i, prompt in enumerate(prompts):
                await batch_progress.update(
                    content=f"""🎨 **Génération Batch en Cours**

**Image {i+1}/{len(prompts)}**
**Prompt** : "{prompt[:50]}..."
**⏳ Statut** : Génération en cours..."""
                )
                
                result = await comfyui_service.generate_image(
                    prompt=prompt,
                    workflow=workflow,
                    style=style,
                    use_cache=True
                )
                
                results.append((prompt, result))
                
                # Pause entre générations pour éviter la surcharge
                if i < len(prompts) - 1:
                    await asyncio.sleep(2)
            
            # Afficher tous les résultats
            await self._display_batch_results(results, batch_progress)
            
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur génération batch** : {str(e)}"
            ).send()
    
    async def _display_batch_results(self, results: List[Tuple[str, Dict[str, Any]]], progress_message: cl.Message):
        """Affiche les résultats d'une génération batch"""
        
        successful_results = [r for r in results if r[1]["success"]]
        failed_results = [r for r in results if not r[1]["success"]]
        
        # Résumé du batch
        summary_content = f"""
🎯 **Génération Batch Terminée**

**✅ Réussies** : {len(successful_results)}/{len(results)}
**❌ Échouées** : {len(failed_results)}/{len(results)}
**📊 Taux de succès** : {len(successful_results)/len(results)*100:.1f}%

---
"""
        
        await progress_message.update(content=summary_content)
        
        # Afficher les images réussies
        if successful_results:
            image_elements = []
            
            for i, (prompt, result) in enumerate(successful_results):
                if result.get("image_data"):
                    image_element = cl.Image(
                        name=f"batch_image_{i+1}.png",
                        content=result["image_data"],
                        display="inline"
                    )
                    image_elements.append(image_element)
            
            if image_elements:
                await cl.Message(
                    content=f"🖼️ **Images Générées** ({len(image_elements)}/{len(results)})",
                    elements=image_elements
                ).send()
        
        # Afficher les erreurs
        if failed_results:
            error_details = "❌ **Images Échouées** :\n\n"
            for prompt, result in failed_results:
                error_details += f"- \"{prompt[:50]}...\" : {result.get('error', 'Erreur inconnue')}\n"
            
            await cl.Message(content=error_details).send()
    
    async def show_generation_gallery(self):
        """Affiche une galerie d'exemples de générations"""
        
        gallery_content = """
🖼️ **Galerie d'Exemples - ComfyUI**

## 🎨 Inspirez-vous de ces exemples

### 📸 Réaliste
- "Portrait d'une femme aux cheveux roux, éclairage naturel, photo studio"
- "Paysage de montagne avec lac, reflets dorés, coucher de soleil"
- "Architecture moderne, lignes épurées, perspective urbaine"

### 🎭 Artistique  
- "Chat cosmique aux yeux étoilés, art numérique surréaliste"
- "Forêt enchantée avec lumières magiques, style fantasy"
- "Robot steampunk dans un atelier vintage, détails mécaniques"

### 🚀 Science-Fiction
- "Vaisseau spatial futuriste, néons bleus, environnement spatial"
- "Cityscape cyberpunk, pluie néon, ambiance nocturne"
- "Armure high-tech, matériaux réfléchissants, design futuriste"

### ⚪ Minimaliste
- "Logo simple et élégant, formes géométriques, palette monochrome" 
- "Architecture épurée, lignes nettes, espaces vides"
- "Nature morte minimaliste, objets simples, éclairage doux"

## 💡 Conseils pour Vos Prompts

**✅ Bon prompt** :
- Spécifique et descriptif
- Mentionne le style souhaité
- Inclut des détails sur l'éclairage
- Précise les couleurs importantes

**❌ À éviter** :
- Descriptions trop vagues
- Prompts contradictoires
- Texte trop long (>200 mots)
- Demandes impossibles

**🚀 Commencez par un prompt simple et ajoutez progressivement des détails !**
"""
        
        await cl.Message(content=gallery_content).send()

# Instance globale
image_gen_interface = ImageGenerationInterface()
