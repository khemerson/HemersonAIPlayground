# app/components/vision_interface.py
"""
Interface Chainlit pour analyse d'images avec Pixtral
Optimisée pour UX et gestion avancée des images
"""

import chainlit as cl
from typing import Dict, Any, Optional, List
import time
import asyncio
from PIL import Image
import io

from services.vision.pixtral_service import pixtral_service

class VisionInterface:
    """Interface Chainlit pour l'analyse d'images"""
    
    def __init__(self):
        self.supported_analysis_types = {
            "comprehensive": {
                "name": "📊 Analyse Complète",
                "description": "Analyse détaillée et structurée",
                "icon": "📊"
            },
            "technical": {
                "name": "🔧 Analyse Technique", 
                "description": "Focus sur les aspects techniques",
                "icon": "🔧"
            },
            "creative": {
                "name": "🎨 Analyse Créative",
                "description": "Approche artistique et créative", 
                "icon": "🎨"
            },
            "document": {
                "name": "📄 Analyse Document",
                "description": "Extraction d'informations textuelles",
                "icon": "📄"
            }
        }
    
    async def handle_image_upload(self, 
                                 image_element: cl.Image, 
                                 user_prompt: str = None,
                                 analysis_type: str = "comprehensive"):
        """
        Traite une image uploadée avec interface progressive
        
        Args:
            image_element: Élément image Chainlit
            user_prompt: Prompt utilisateur personnalisé
            analysis_type: Type d'analyse à effectuer
        """
        try:
            # Validation préliminaire
            if not self._validate_image_element(image_element):
                await cl.Message(
                    content="❌ **Image invalide**\n\nFormats supportés : JPEG, PNG, WEBP, GIF, BMP\nTaille max : 25MB\nRésolution max recommandée : 2048x2048"
                ).send()
                return
            
            # Déterminer le prompt d'analyse
            if not user_prompt:
                user_prompt = self._generate_default_prompt(analysis_type, image_element.name)
            
            # Interface de progression avec étapes détaillées
            progress_message = await cl.Message(
                content=f"🖼️ **Analyse d'Image Démarrée**\n\n**Image** : {image_element.name}\n**Type** : {self.supported_analysis_types[analysis_type]['name']}\n**Statut** : Initialisation..."
            ).send()
            
            # Étape 1 : Préparation
            await progress_message.update(
                content=f"🖼️ **Analyse d'Image en Cours**\n\n**Image** : {image_element.name}\n**Type** : {self.supported_analysis_types[analysis_type]['name']}\n\n1️⃣ **Préparation** : Validation et optimisation...\n2️⃣ **Analyse** : En attente\n3️⃣ **Finalisation** : En attente"
            )
            
            # Étape 2 : Analyse avec Pixtral
            await progress_message.update(
                content=f"🖼️ **Analyse d'Image en Cours**\n\n**Image** : {image_element.name}\n**Type** : {self.supported_analysis_types[analysis_type]['name']}\n\n1️⃣ **Préparation** : ✅ Terminée\n2️⃣ **Analyse** : 🔄 Analyse Pixtral en cours...\n3️⃣ **Finalisation** : En attente"
            )
            
            # Lancer l'analyse Pixtral
            result = await pixtral_service.analyze_image(
                image_data=image_element.content,
                filename=image_element.name,
                prompt=user_prompt,
                analysis_type=analysis_type,
                optimization_level="balanced"
            )
            
            if result["success"]:
                await self._display_successful_analysis(result, progress_message, image_element)
            else:
                await self._display_analysis_error(result, progress_message, image_element)
                
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur Analyse d'Image**\n\n{str(e)}\n\nVeuillez réessayer avec une autre image."
            ).send()
    
    async def _display_successful_analysis(self, 
                                          result: Dict[str, Any], 
                                          progress_message: cl.Message,
                                          image_element: cl.Image):
        """Affiche le résultat d'une analyse réussie"""
        
        # Extraire les informations
        analysis = result["analysis"]
        metadata = result["metadata"]
        processing_times = result["processing_times"]
        
        # Message de succès avec analyse complète
        success_content = f"""
🎉 **Analyse d'Image Terminée avec Succès !**

## 🖼️ Image Analysée
**Fichier** : {metadata['filename']}
**Résolution originale** : {metadata['image_metadata']['original_size'][0]}x{metadata['image_metadata']['original_size'][1]}
**Format** : {metadata['image_metadata']['original_format']}
**Taille** : {metadata['image_metadata']['original_bytes'] / 1024:.1f} KB

## 🧠 Analyse Pixtral ({result['analysis_type']})

{analysis}

---

## ⚡ Métriques de Performance
**Temps de préparation** : {processing_times['preprocessing']:.2f}s
**Temps d'analyse** : {processing_times['analysis']:.2f}s
**Temps total** : {processing_times['total']:.2f}s
**Tokens utilisés** : {metadata['tokens_used']}
**Qualité d'analyse** : {metadata['analysis_quality']['quality_label'].title()}

## 🔧 Informations Techniques
**Modèle** : {result['model']}
**GPU** : {result['gpu']}
**Optimisation** : {metadata['image_metadata']['optimization_level'].title()}
**Ratio compression** : {metadata['image_metadata'].get('compression_ratio', 1.0):.1f}x
"""
        
        # Mettre à jour le message de progression
        await progress_message.update(content=success_content)
        
        # Proposer des analyses supplémentaires
        if result["analysis_type"].lower() != "comprehensive":
            await self._suggest_additional_analyses(image_element, result["analysis_type"])
    
    async def _display_analysis_error(self, 
                                     result: Dict[str, Any], 
                                     progress_message: cl.Message,
                                     image_element: cl.Image):
        """Affiche les erreurs d'analyse avec diagnostic"""
        
        processing_times = result.get("processing_times", {})
        
        error_content = f"""
❌ **Erreur Analyse d'Image**

**Image** : {image_element.name}
**Erreur** : {result.get('error', 'Erreur inconnue')}
**Temps écoulé** : {processing_times.get('total', 0):.2f}s

## 🔍 Diagnostic
L'analyse d'image a échoué. Causes possibles :
- Service Pixtral temporairement indisponible
- Image dans un format non optimal
- Contenu d'image trop complexe pour l'analyse
- Problème de réseau temporaire

**🔧 Solutions** :
1. **Réessayez** dans quelques instants
2. **Optimisez l'image** : réduisez la taille si > 10MB
3. **Changez le format** : préférez JPEG ou PNG
4. **Simplifiez le prompt** si très spécialisé

**💡 Conseil** : Les images nettes avec un bon contraste donnent de meilleurs résultats.
"""
        
        await progress_message.update(content=error_content)
    
    async def _suggest_additional_analyses(self, image_element: cl.Image, current_type: str):
        """Suggère des analyses complémentaires"""
        
        # Analyses complémentaires suggérées
        suggestions = []
        for analysis_type, info in self.supported_analysis_types.items():
            if analysis_type != current_type.lower():
                suggestions.append(
                    cl.Action(
                        name=f"analyze_{analysis_type}",
                        label=info["name"],
                        description=info["description"],
                        icon=info["icon"]
                    )
                )
        
        if suggestions:
            await cl.Message(
                content="💡 **Analyses Complémentaires Disponibles**\n\nVoulez-vous analyser cette image sous un autre angle ?",
                actions=suggestions[:3]  # Limiter à 3 suggestions
            ).send()
    
    async def handle_multi_image_analysis(self, image_elements: List[cl.Image], comparison_prompt: str = None):
        """
        Analyse et comparaison de multiple images
        
        Args:
            image_elements: Liste d'éléments images
            comparison_prompt: Prompt de comparaison
        """
        try:
            if len(image_elements) > 4:
                await cl.Message(
                    content="❌ **Trop d'images**\n\nMaximum 4 images simultanément pour l'analyse comparative."
                ).send()
                return
            
            # Prompt par défaut pour comparaison
            if not comparison_prompt:
                comparison_prompt = "Compare ces images et identifie les similitudes, différences et éléments remarquables."
            
            # Message de progression
            progress_message = await cl.Message(
                content=f"🖼️ **Analyse Multi-Images**\n\n**Images** : {len(image_elements)}\n**Statut** : Analyse en cours..."
            ).send()
            
            # Analyser chaque image en parallèle
            analysis_tasks = []
            for i, img_element in enumerate(image_elements):
                task = pixtral_service.analyze_image(
                    image_data=img_element.content,
                    filename=f"{img_element.name} (Image {i+1})",
                    prompt=f"Image {i+1}/|{len(image_elements)}: {comparison_prompt}",
                    analysis_type="comprehensive"
                )
                analysis_tasks.append(task)
            
            # Exécuter les analyses en parallèle
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Traiter les résultats
            successful_analyses = []
            failed_analyses = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_analyses.append(f"Image {i+1}: {str(result)}")
                elif result.get("success", False):
                    successful_analyses.append({
                        "index": i+1,
                        "filename": image_elements[i].name,
                        "analysis": result["analysis"]
                    })
                else:
                    failed_analyses.append(f"Image {i+1}: {result.get('error', 'Erreur inconnue')}")
            
            # Construire le rapport comparatif
            if successful_analyses:
                comparison_report = self._build_comparison_report(successful_analyses, comparison_prompt)
                await progress_message.update(content=comparison_report)
            else:
                await progress_message.update(
                    content="❌ **Échec Analyse Multi-Images**\n\nAucune image n'a pu être analysée avec succès."
                )
                
        except Exception as e:
            await cl.Message(
                content=f"❌ **Erreur Analyse Multi-Images**\n\n{str(e)}"
            ).send()
    
    def _build_comparison_report(self, analyses: List[Dict], prompt: str) -> str:
        """Construit un rapport comparatif des analyses"""
        
        report = f"""
🎯 **Analyse Multi-Images Terminée**

**Prompt de comparaison** : {prompt}
**Images analysées** : {len(analyses)}

---

"""
        
        # Analyses individuelles
        for analysis in analyses:
            report += f"""
## 🖼️ Image {analysis['index']} - {analysis['filename']}

{analysis['analysis']}

---
"""
        
        # Synthèse comparative (basique)
        report += f"""
## 📊 Synthèse Comparative

**Nombre d'images traitées** : {len(analyses)}
**Analyses réussies** : {len(analyses)}
**Modèle utilisé** : Pixtral 12B

💡 **Note** : Pour des comparaisons plus approfondies, utilisez l'analyse individuelle de chaque image puis demandez une synthèse manuelle.
"""
        
        return report
    
    def _validate_image_element(self, image_element: cl.Image) -> bool:
        """Valide un élément image Chainlit"""
        try:
            # Vérifier la présence du contenu
            if not hasattr(image_element, 'content') or not image_element.content:
                return False
            
            # Vérifier la taille
            if len(image_element.content) > 25 * 1024 * 1024:  # 25MB
                return False
            
            # Vérifier qu'on peut ouvrir l'image
            try:
                with Image.open(io.BytesIO(image_element.content)) as img:
                    # Vérification basique
                    if img.size[0] < 10 or img.size[1] < 10:  # Trop petite
                        return False
                    return True
            except:
                return False
                
        except Exception:
            return False
    
    def _generate_default_prompt(self, analysis_type: str, filename: str) -> str:
        """Génère un prompt par défaut selon le type d'analyse"""
        
        default_prompts = {
            "comprehensive": f"Effectue une analyse complète et détaillée de cette image ({filename}). Décris tous les éléments visibles, la composition, les couleurs, le style et le contexte.",
            
            "technical": f"Analyse technique de cette image ({filename}): qualité, résolution, technique de création, aspects techniques remarquables, et suggestions d'optimisation.",
            
            "creative": f"Analyse créative et artistique de cette image ({filename}): style artistique, créativité, composition esthétique, inspiration et potentiel créatif.",
            
            "document": f"Analyse ce document ({filename}) pour extraire toutes les informations textuelles visibles, la structure du document et les éléments informatifs importants."
        }
        
        return default_prompts.get(analysis_type, default_prompts["comprehensive"])

# Instance globale
vision_interface = VisionInterface()
