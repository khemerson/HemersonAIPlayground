# app/services/image_generation/comfyui_service.py
"""
Service de génération d'images avec ComfyUI - HemersonAIBuild PlaygroundV1
Intégration optimisée pour Chainlit 2.5.5 avec gestion avancée des workflows
"""

import asyncio
import time
import json
import base64
import hashlib
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import tempfile
from PIL import Image, ImageEnhance, ImageFilter
import io
import aiofiles
import httpx
from loguru import logger

from services.rate_limiter.advanced_limiter import CacheManager, RequestType
from config.settings import settings

class ComfyUIService:
    """
    Service de génération d'images avec ComfyUI - Production Ready
    
    Fonctionnalités avancées :
    - Workflows prédéfinis optimisés
    - Génération batch et single
    - Post-processing intelligent
    - Cache des générations
    - Queue management avancé
    - Templates de prompts spécialisés
    """
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self.comfyui_endpoint = settings.COMFYUI_ENDPOINT
        
        # Client HTTP optimisé pour génération d'images
        self.comfy_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=20.0,
                read=600.0,  # 10 minutes pour génération complexe
                write=300.0,
                pool=720.0
            ),
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=20,
                keepalive_expiry=300.0
            )
        )
        
        # Configuration des workflows
        self.workflows = self._initialize_workflows()
        
        # Templates de prompts spécialisés
        self.prompt_templates = self._initialize_prompt_templates()
        
        # Métriques de génération
        self.generation_metrics = {
            "images_generated": 0,
            "successful_generations": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "cache_hits": 0,
            "workflow_usage": {},
            "popular_styles": {}
        }
        
        logger.info("🎨 ComfyUIService initialisé avec workflows optimisés")
    
    def _initialize_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les workflows ComfyUI prédéfinis"""
        return {
            "standard": {
                "name": "Génération Standard",
                "description": "Workflow équilibré qualité/vitesse",
                "steps": 20,
                "cfg_scale": 7.5,
                "sampler": "euler_a",
                "scheduler": "normal",
                "width": 1024,
                "height": 1024,
                "model": "stable-diffusion-xl-base",
                "vae": "sdxl_vae",
                "estimated_time": 30
            },
            "quality": {
                "name": "Haute Qualité",
                "description": "Maximum de qualité, temps plus long",
                "steps": 35,
                "cfg_scale": 8.0,
                "sampler": "dpmpp_2m",
                "scheduler": "karras",
                "width": 1280,
                "height": 1280,
                "model": "stable-diffusion-xl-refiner",
                "vae": "sdxl_vae",
                "estimated_time": 60
            },
            "speed": {
                "name": "Génération Rapide",
                "description": "Optimisé pour la vitesse",
                "steps": 12,
                "cfg_scale": 6.5,
                "sampler": "euler",
                "scheduler": "simple",
                "width": 768,
                "height": 768,
                "model": "stable-diffusion-turbo",
                "vae": "sd_vae",
                "estimated_time": 15
            },
            "artistic": {
                "name": "Style Artistique",
                "description": "Focus créativité et style",
                "steps": 25,
                "cfg_scale": 9.0,
                "sampler": "dpmpp_2m_sde",
                "scheduler": "exponential",
                "width": 1024,
                "height": 1152,
                "model": "stable-diffusion-xl-base",
                "vae": "sdxl_vae",
                "estimated_time": 45
            },
            "portrait": {
                "name": "Portrait Optimisé",
                "description": "Spécialisé pour les portraits",
                "steps": 28,
                "cfg_scale": 7.0,
                "sampler": "dpmpp_2m",
                "scheduler": "normal",
                "width": 832,
                "height": 1216,
                "model": "stable-diffusion-xl-base",
                "vae": "sdxl_vae",
                "estimated_time": 40
            }
        }
    
    def _initialize_prompt_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialise les templates de prompts spécialisés"""
        return {
            "realistic": {
                "name": "Photo Réaliste",
                "prefix": "realistic photography, high quality, detailed, professional lighting",
                "suffix": "8k resolution, sharp focus, realistic textures, professional photography",
                "negative": "cartoon, anime, painting, sketch, blurry, low quality, distorted"
            },
            "artistic": {
                "name": "Art Numérique",
                "prefix": "digital art, artistic composition, creative, vibrant colors",
                "suffix": "masterpiece, highly detailed, creative composition, artistic style",
                "negative": "photography, realistic, blurry, low quality, amateur"
            },
            "fantasy": {
                "name": "Fantasy/Fantastique",
                "prefix": "fantasy art, magical, mystical, ethereal, enchanted",
                "suffix": "fantasy style, magical atmosphere, detailed fantasy art, mystical lighting",
                "negative": "modern, contemporary, realistic photography, mundane, ordinary"
            },
            "scifi": {
                "name": "Science-Fiction",
                "prefix": "sci-fi, futuristic, technological, advanced, cyberpunk",
                "suffix": "futuristic design, high-tech, science fiction style, advanced technology",
                "negative": "medieval, ancient, primitive, low-tech, historical"
            },
            "minimalist": {
                "name": "Minimaliste",
                "prefix": "minimalist design, clean, simple, elegant, refined",
                "suffix": "minimalist style, clean composition, simple elegance, refined aesthetics",
                "negative": "cluttered, busy, complex, ornate, excessive detail"
            }
        }
    
    async def generate_image(self,
                           prompt: str,
                           workflow: str = "standard",
                           style: str = "artistic",
                           use_cache: bool = True,
                           **kwargs) -> Dict[str, Any]:
        """
        Génération d'image principale avec ComfyUI
        
        Args:
            prompt: Prompt de génération
            workflow: Type de workflow (standard, quality, speed, artistic, portrait)
            style: Style de template (realistic, artistic, fantasy, scifi, minimalist)
            use_cache: Utiliser le cache intelligent
            **kwargs: Paramètres personnalisés
            
        Returns:
            Image générée avec métadonnées complètes
        """
        generation_start = time.time()
        self.generation_metrics["images_generated"] += 1
        
        try:
            logger.info(f"🎨 Début génération: {prompt[:50]}... (workflow: {workflow}, style: {style})")
            
            # Vérifier le cache
            if use_cache:
                cache_key = self._generate_cache_key(prompt, workflow, style, kwargs)
                
                cached_result = await self.cache_manager.get_cached_response(cache_key)
                if cached_result:
                    self.generation_metrics["cache_hits"] += 1
                    logger.info(f"💾 Génération depuis cache: {workflow}/{style}")
                    return cached_result
            
            # Construire le prompt enrichi
            enhanced_prompt = self._build_enhanced_prompt(prompt, style, kwargs)
            
            # Préparer la configuration du workflow
            workflow_config = self._prepare_workflow_config(workflow, enhanced_prompt, kwargs)
            
            # Générer l'image
            result = await self._execute_comfyui_workflow(workflow_config)
            
            if result["success"]:
                # Post-processing si demandé
                if kwargs.get("post_process", False):
                    result = await self._apply_post_processing(result, kwargs)
                
                # Construire la réponse complète
                generation_time = time.time() - generation_start
                
                complete_result = {
                    "success": True,
                    "image_data": result["image_data"],
                    "image_url": result.get("image_url", ""),
                    "image_info": {
                        "width": workflow_config["width"],
                        "height": workflow_config["height"],
                        "format": "PNG",
                        "size_bytes": len(result["image_data"]) if result["image_data"] else 0
                    },
                    "generation_params": {
                        "prompt": prompt,
                        "enhanced_prompt": enhanced_prompt["full_prompt"],
                        "negative_prompt": enhanced_prompt["negative_prompt"],
                        "workflow": workflow,
                        "style": style,
                        "steps": workflow_config["steps"],
                        "cfg_scale": workflow_config["cfg_scale"],
                        "sampler": workflow_config["sampler"],
                        "seed": workflow_config.get("seed", -1)
                    },
                    "metadata": {
                        "model": "ComfyUI SDXL",
                        "gpu": "RTX 4090 (192.168.1.213)",
                        "workflow_name": self.workflows[workflow]["name"],
                        "style_name": self.prompt_templates[style]["name"],
                        "generation_time": generation_time,
                        "estimated_vs_actual": {
                            "estimated": self.workflows[workflow]["estimated_time"],
                            "actual": generation_time,
                            "efficiency": self.workflows[workflow]["estimated_time"] / generation_time
                        },
                        "quality_score": self._calculate_quality_score(workflow_config),
                        "cached": False
                    }
                }
                
                # Mettre en cache
                if use_cache:
                    cache_ttl = 14400  # 4 heures pour les générations (coûteuses)
                    await self.cache_manager.cache_response(
                        cache_key,
                        complete_result,
                        ttl=cache_ttl
                    )
                
                # Mettre à jour les métriques
                self._update_generation_metrics(True, generation_time, workflow, style)
                self.generation_metrics["successful_generations"] += 1
                
                logger.info(f"✅ Génération ComfyUI réussie en {generation_time:.2f}s")
                
                return complete_result
            else:
                self._update_generation_metrics(False, time.time() - generation_start, workflow, style)
                return {
                    "success": False,
                    "error": result["error"],
                    "image_data": None,
                    "model": "ComfyUI SDXL",
                    "generation_time": time.time() - generation_start
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur génération ComfyUI: {e}")
            return {
                "success": False,
                "error": str(e),
                "image_data": None,
                "model": "ComfyUI SDXL",
                "generation_time": time.time() - generation_start
            }
    
    def _build_enhanced_prompt(self, prompt: str, style: str, params: Dict) -> Dict[str, str]:
        """Construit un prompt enrichi avec le style sélectionné"""
        try:
            template = self.prompt_templates.get(style, self.prompt_templates["artistic"])
            
            # Construire le prompt complet
            full_prompt_parts = []
            
            # Prefix du style
            if template["prefix"]:
                full_prompt_parts.append(template["prefix"])
            
            # Prompt utilisateur (cœur)
            full_prompt_parts.append(prompt.strip())
            
            # Suffix du style
            if template["suffix"]:
                full_prompt_parts.append(template["suffix"])
            
            # Qualificateurs additionnels
            quality_boost = params.get("quality_boost", True)
            if quality_boost:
                full_prompt_parts.append("high quality, detailed, masterpiece")
            
            full_prompt = ", ".join(full_prompt_parts)
            
            # Prompt négatif
            negative_parts = [template["negative"]]
            if params.get("extra_negative"):
                negative_parts.append(params["extra_negative"])
            
            negative_prompt = ", ".join(negative_parts)
            
            return {
                "full_prompt": full_prompt,
                "negative_prompt": negative_prompt,
                "original_prompt": prompt,
                "style_applied": style
            }
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur construction prompt: {e}")
            return {
                "full_prompt": prompt,
                "negative_prompt": "low quality, blurry, distorted",
                "original_prompt": prompt,
                "style_applied": "fallback"
            }
    
    def _prepare_workflow_config(self, workflow: str, enhanced_prompt: Dict, params: Dict) -> Dict[str, Any]:
        """Prépare la configuration du workflow ComfyUI"""
        base_config = self.workflows.get(workflow, self.workflows["standard"]).copy()
        
        # Appliquer les overrides personnalisés
        overrides = {
            "width": params.get("width", base_config["width"]),
            "height": params.get("height", base_config["height"]),
            "steps": params.get("steps", base_config["steps"]),
            "cfg_scale": params.get("cfg_scale", base_config["cfg_scale"]),
            "sampler": params.get("sampler", base_config["sampler"]),
            "seed": params.get("seed", -1),
            "batch_size": params.get("batch_size", 1)
        }
        
        # Fusionner avec la config de base
        workflow_config = {**base_config, **overrides}
        
        # Ajouter les prompts
        workflow_config.update({
            "positive_prompt": enhanced_prompt["full_prompt"],
            "negative_prompt": enhanced_prompt["negative_prompt"]
        })
        
        # Validation des paramètres
        workflow_config = self._validate_workflow_params(workflow_config)
        
        return workflow_config
    
    def _validate_workflow_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Valide et corrige les paramètres du workflow"""
        # Limites de résolution
        config["width"] = max(512, min(2048, config["width"]))
        config["height"] = max(512, min(2048, config["height"]))
        
        # Assurer dimensions multiples de 64 (requis SDXL)
        config["width"] = (config["width"] // 64) * 64
        config["height"] = (config["height"] // 64) * 64
        
        # Limites des steps
        config["steps"] = max(8, min(100, config["steps"]))
        
        # Limites CFG scale
        config["cfg_scale"] = max(1.0, min(20.0, config["cfg_scale"]))
        
        # Batch size
        config["batch_size"] = max(1, min(4, config["batch_size"]))
        
        return config
    
    async def _execute_comfyui_workflow(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Exécute le workflow ComfyUI avec gestion de queue"""
        try:
            # Construire le payload ComfyUI
            workflow_payload = self._build_comfyui_payload(config)
            
            # Soumettre à la queue
            queue_response = await self.comfy_client.post(
                f"{self.comfyui_endpoint}/prompt",
                json={"prompt": workflow_payload}
            )
            
            if queue_response.status_code != 200:
                return {
                    "success": False,
                    "error": f"Erreur soumission queue: {queue_response.status_code}"
                }
            
            queue_data = queue_response.json()
            prompt_id = queue_data.get("prompt_id")
            
            if not prompt_id:
                return {
                    "success": False,
                    "error": "Pas de prompt_id reçu"
                }
            
            logger.info(f"🎯 Workflow soumis à ComfyUI: {prompt_id}")
            
            # Attendre la completion
            result = await self._wait_for_completion(prompt_id, config["steps"])
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Erreur exécution workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _build_comfyui_payload(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Construit le payload ComfyUI optimisé"""
        # Template de workflow simplifié mais complet
        workflow = {
            "1": {
                "inputs": {
                    "ckpt_name": config["model"],
                },
                "class_type": "CheckpointLoaderSimple"
            },
            "2": {
                "inputs": {
                    "text": config["positive_prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "3": {
                "inputs": {
                    "text": config["negative_prompt"],
                    "clip": ["1", 1]
                },
                "class_type": "CLIPTextEncode"
            },
            "4": {
                "inputs": {
                    "width": config["width"],
                    "height": config["height"],
                    "batch_size": config["batch_size"]
                },
                "class_type": "EmptyLatentImage"
            },
            "5": {
                "inputs": {
                    "seed": config["seed"],
                    "steps": config["steps"],
                    "cfg": config["cfg_scale"],
                    "sampler_name": config["sampler"],
                    "scheduler": config["scheduler"],
                    "denoise": 1.0,
                    "model": ["1", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0]
                },
                "class_type": "KSampler"
            },
            "6": {
                "inputs": {
                    "samples": ["5", 0],
                    "vae": ["1", 2]
                },
                "class_type": "VAEDecode"
            },
            "7": {
                "inputs": {
                    "filename_prefix": "PlaygroundV1_",
                    "images": ["6", 0]
                },
                "class_type": "SaveImage"
            }
        }
        
        return workflow
    
    async def _wait_for_completion(self, prompt_id: str, steps: int) -> Dict[str, Any]:
        """Attendre la completion du workflow avec timeout adaptatif"""
        # Timeout basé sur le nombre de steps
        base_timeout = 30
        step_timeout = steps * 2  # 2 secondes par step
        max_timeout = min(600, base_timeout + step_timeout)  # Maximum 10 minutes
        
        start_time = time.time()
        check_interval = 3  # Vérifier toutes les 3 secondes
        
        while time.time() - start_time < max_timeout:
            try:
                # Vérifier le statut
                history_response = await self.comfy_client.get(
                    f"{self.comfyui_endpoint}/history/{prompt_id}"
                )
                
                if history_response.status_code == 200:
                    history_data = history_response.json()
                    
                    if prompt_id in history_data:
                        # Workflow terminé
                        result_data = history_data[prompt_id]
                        
                        if "outputs" in result_data:
                            # Récupérer l'image générée
                            image_data = await self._extract_generated_image(result_data["outputs"])
                            
                            if image_data:
                                return {
                                    "success": True,
                                    "image_data": image_data,
                                    "generation_info": result_data.get("status", {}),
                                    "prompt_id": prompt_id
                                }
                            else:
                                return {
                                    "success": False,
                                    "error": "Image non trouvée dans la réponse"
                                }
                
                # Attendre avant la prochaine vérification
                await asyncio.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur vérification statut: {e}")
                await asyncio.sleep(check_interval)
        
        return {
            "success": False,
            "error": f"Timeout après {max_timeout}s"
        }
    
    async def _extract_generated_image(self, outputs: Dict[str, Any]) -> Optional[bytes]:
        """Extrait l'image générée des outputs ComfyUI"""
        try:
            # Trouver la node de sauvegarde (généralement la dernière)
            save_nodes = [node for node in outputs.values() if "images" in node]
            
            if not save_nodes:
                logger.error("❌ Aucune node de sauvegarde trouvée")
                return None
            
            # Prendre la première image de la première node
            save_node = save_nodes[0]
            if not save_node["images"]:
                logger.error("❌ Aucune image dans la node de sauvegarde")
                return None
            
            image_info = save_node["images"][0]
            filename = image_info["filename"]
            subfolder = image_info.get("subfolder", "")
            
            # Construire l'URL de l'image
            image_path = f"/view?filename={filename}"
            if subfolder:
                image_path += f"&subfolder={subfolder}"
            
            # Télécharger l'image
            image_response = await self.comfy_client.get(
                f"{self.comfyui_endpoint}{image_path}"
            )
            
            if image_response.status_code == 200:
                return image_response.content
            else:
                logger.error(f"❌ Erreur téléchargement image: {image_response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Erreur extraction image: {e}")
            return None
    
    async def _apply_post_processing(self, result: Dict[str, Any], params: Dict) -> Dict[str, Any]:
        """Applique du post-processing sur l'image générée"""
        try:
            if not result.get("image_data"):
                return result
            
            # Charger l'image
            image = Image.open(io.BytesIO(result["image_data"]))
            
            # Appliquer les améliorations demandées
            if params.get("enhance_sharpness"):
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(params.get("sharpness_factor", 1.2))
            
            if params.get("enhance_contrast"):
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(params.get("contrast_factor", 1.1))
            
            if params.get("enhance_color"):
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(params.get("color_factor", 1.1))
            
            if params.get("apply_filter"):
                filter_type = params.get("filter_type", "SHARPEN")
                if hasattr(ImageFilter, filter_type):
                    image = image.filter(getattr(ImageFilter, filter_type))
            
            # Sauvegarder l'image modifiée
            output_buffer = io.BytesIO()
            image.save(output_buffer, format='PNG', quality=95, optimize=True)
            
            result["image_data"] = output_buffer.getvalue()
            result["post_processed"] = True
            result["post_processing_applied"] = list(params.keys())
            
            logger.info("✅ Post-processing appliqué avec succès")
            
            return result
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur post-processing: {e}")
            # Retourner le résultat original en cas d'erreur
            return result
    
    def _generate_cache_key(self, prompt: str, workflow: str, style: str, params: Dict) -> str:
        """Génère une clé de cache unique pour la génération"""
        # Paramètres significatifs pour le cache
        cache_params = {
            "prompt": prompt,
            "workflow": workflow,
            "style": style,
            "width": params.get("width"),
            "height": params.get("height"),
            "steps": params.get("steps"),
            "cfg_scale": params.get("cfg_scale"),
            "seed": params.get("seed", -1)
        }
        
        # Hash des paramètres
        cache_string = json.dumps(cache_params, sort_keys=True)
        cache_hash = hashlib.md5(cache_string.encode()).hexdigest()[:16]
        
        return f"comfyui_generation:{cache_hash}"
    
    def _calculate_quality_score(self, config: Dict[str, Any]) -> float:
        """Calcule un score de qualité basé sur la configuration"""
        try:
            # Facteurs de qualité
            resolution_factor = (config["width"] * config["height"]) / (1024 * 1024)  # Normalisé sur 1024x1024
            steps_factor = min(config["steps"] / 30, 1.0)  # Optimal autour de 30 steps
            cfg_factor = 1.0 - abs(config["cfg_scale"] - 7.5) / 10.0  # Optimal autour de 7.5
            
            # Score combiné
            quality_score = (resolution_factor * 0.4 + steps_factor * 0.4 + cfg_factor * 0.2)
            
            return max(0.1, min(1.0, quality_score))
            
        except Exception:
            return 0.5
    
    def _update_generation_metrics(self, success: bool, generation_time: float, workflow: str, style: str):
        """Met à jour les métriques de génération"""
        try:
            # Temps de génération
            self.generation_metrics["total_generation_time"] += generation_time
            
            if self.generation_metrics["successful_generations"] > 0:
                self.generation_metrics["average_generation_time"] = (
                    self.generation_metrics["total_generation_time"] / 
                    self.generation_metrics["successful_generations"]
                )
            
            # Usage des workflows
            if workflow not in self.generation_metrics["workflow_usage"]:
                self.generation_metrics["workflow_usage"][workflow] = 0
            self.generation_metrics["workflow_usage"][workflow] += 1
            
            # Styles populaires
            if style not in self.generation_metrics["popular_styles"]:
                self.generation_metrics["popular_styles"][style] = 0
            self.generation_metrics["popular_styles"][style] += 1
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur mise à jour métriques: {e}")
    
    async def get_available_models(self) -> Dict[str, Any]:
        """Récupère la liste des modèles disponibles sur ComfyUI"""
        try:
            response = await self.comfy_client.get(f"{self.comfyui_endpoint}/object_info")
            
            if response.status_code == 200:
                object_info = response.json()
                
                # Extraire les modèles disponibles
                checkpoints = object_info.get("CheckpointLoaderSimple", {}).get("input", {}).get("required", {}).get("ckpt_name", [[]])[0]
                samplers = object_info.get("KSampler", {}).get("input", {}).get("required", {}).get("sampler_name", [[]])[0]
                schedulers = object_info.get("KSampler", {}).get("input", {}).get("required", {}).get("scheduler", [[]])[0]
                
                return {
                    "success": True,
                    "models": {
                        "checkpoints": checkpoints,
                        "samplers": samplers,
                        "schedulers": schedulers
                    },
                    "workflows": list(self.workflows.keys()),
                    "styles": list(self.prompt_templates.keys())
                }
            else:
                return {
                    "success": False,
                    "error": f"Erreur récupération modèles: {response.status_code}"
                }
                
        except Exception as e:
            logger.error(f"❌ Erreur récupération modèles: {e}")
            return {
                "success": False,
                "error": str(e),
                "models": {
                    "checkpoints": ["stable-diffusion-xl-base"],
                    "samplers": ["euler_a", "dpmpp_2m"],
                    "schedulers": ["normal", "karras"]
                },
                "workflows": list(self.workflows.keys()),
                "styles": list(self.prompt_templates.keys())
            }
    
    async def get_generation_metrics(self) -> Dict[str, Any]:
        """Retourne les métriques de génération détaillées"""
        try:
            success_rate = 0
            if self.generation_metrics["images_generated"] > 0:
                success_rate = (
                    self.generation_metrics["successful_generations"] / 
                    self.generation_metrics["images_generated"]
                ) * 100
            
            cache_hit_rate = 0
            if self.generation_metrics["images_generated"] > 0:
                cache_hit_rate = (
                    self.generation_metrics["cache_hits"] / 
                    self.generation_metrics["images_generated"]
                ) * 100
            
            # Top workflows et styles
            top_workflow = max(self.generation_metrics["workflow_usage"], 
                             key=self.generation_metrics["workflow_usage"].get) if self.generation_metrics["workflow_usage"] else "standard"
            
            top_style = max(self.generation_metrics["popular_styles"], 
                          key=self.generation_metrics["popular_styles"].get) if self.generation_metrics["popular_styles"] else "artistic"
            
            return {
                **self.generation_metrics,
                "success_rate_percent": success_rate,
                "cache_hit_rate_percent": cache_hit_rate,
                "most_used_workflow": top_workflow,
                "most_popular_style": top_style,
                "efficiency_metrics": {
                    "avg_time_vs_estimated": self._calculate_efficiency_ratio(),
                    "gpu_utilization": "RTX 4090 @ 192.168.1.213"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur métriques génération: {e}")
            return self.generation_metrics
    
    def _calculate_efficiency_ratio(self) -> float:
        """Calcule le ratio d'efficacité temps réel vs estimé"""
        try:
            if not self.generation_metrics["workflow_usage"]:
                return 1.0
            
            # Calculer la moyenne pondérée des ratios d'efficacité
            total_weight = sum(self.generation_metrics["workflow_usage"].values())
            if total_weight == 0:
                return 1.0
            
            efficiency_sum = 0
            for workflow, count in self.generation_metrics["workflow_usage"].items():
                estimated_time = self.workflows.get(workflow, {}).get("estimated_time", 30)
                actual_avg = self.generation_metrics["average_generation_time"]
                if actual_avg > 0:
                    efficiency = estimated_time / actual_avg
                    efficiency_sum += efficiency * count
            
            return efficiency_sum / total_weight
            
        except Exception:
            return 1.0
    
    async def close(self):
        """Fermeture propre du service"""
        await self.comfy_client.aclose()
        logger.info("🔌 ComfyUIService fermé proprement")

# Instance globale
comfyui_service = ComfyUIService(cache_manager)
