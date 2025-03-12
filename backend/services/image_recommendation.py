import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http import models

from backend.models.image import ImageMetadata, ImageRecommendation, ImageRecommendationResponse


class ImageRecommendationService:
    def __init__(self):
        # Initialize CLIP model for image embeddings
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize Qdrant client
        self.qdrant = QdrantClient("localhost", port=6333)
        self.collection_name = "midjourney-images"

        # Create collection if it doesn't exist
        self._initialize_collection()

    def _initialize_collection(self):
        """Initialize Qdrant collection for image features"""
        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # CLIP embedding size
                    distance=models.Distance.COSINE,
                ),
                # No need to create collection as it exists with Midjourney data
                if_exists=models.CollectionExistsAction.IGNORE,
            )

    def _generate_image_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for an image using CLIP"""
        image = Image.open(image_path)
        inputs = self.processor(images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        return image_features.numpy()[0]

    def _extract_image_features(self, image_path: str) -> Dict[str, Any]:
        """Extract or fetch image features"""
        if image_path.startswith(("http://", "https://")):
            # For remote images, return URL-based metadata
            return {
                "type": "remote",
                "url": image_path,
            }
        else:
            # For local images
            image = Image.open(image_path)
            return {
                "type": "local",
                "dimensions": {"width": image.width, "height": image.height},
                "format": image.format,
                "mode": image.mode,
                "size": Path(image_path).stat().st_size,
            }

    def index_image(self, image_path: str, labels: List[str], image_id: str):
        """Index an image in Qdrant"""
        # Generate embedding
        embedding = self._generate_image_embedding(image_path)

        # Extract technical features
        tech_features = self._extract_image_features(image_path)

        # Create metadata
        metadata = ImageMetadata(
            id=image_id,
            filename=Path(image_path).name,
            labels=labels,
            technical_metadata=tech_features,
            content_features={"embedding_size": embedding.shape[0]},
        )

        # Index in Qdrant
        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=hash(image_id), vector=embedding.tolist(), payload=json.loads(metadata.model_dump_json())
                )
            ],
        )

    def _calculate_similarity_aspects(
        self, reference_metadata: Dict[str, Any], candidate_metadata: Dict[str, Any]
    ) -> List[str]:
        """Calculate aspects of similarity between two images"""
        aspects = []

        # For Midjourney images
        if "name" in candidate_metadata:
            aspects.append(f"Artist/Style: {candidate_metadata['name']}")

        if "url" in candidate_metadata:
            style = candidate_metadata["url"].replace("/styles/", "").replace("-", " ").title()
            aspects.append(f"Art Style: {style}")

        # For local images, compare technical aspects if available
        if (
            "technical_metadata" in reference_metadata
            and "technical_metadata" in candidate_metadata
            and all(
                meta.get("type") == "local"
                for meta in [reference_metadata["technical_metadata"], candidate_metadata["technical_metadata"]]
            )
        ):
            ref_tech = reference_metadata["technical_metadata"]
            cand_tech = candidate_metadata["technical_metadata"]

            if abs(ref_tech["dimensions"]["width"] - cand_tech["dimensions"]["width"]) < 100:
                aspects.append("Similar width")
            if abs(ref_tech["dimensions"]["height"] - cand_tech["dimensions"]["height"]) < 100:
                aspects.append("Similar height")
            if ref_tech["format"] == cand_tech["format"]:
                aspects.append(f"Same format: {ref_tech['format']}")

        return aspects or ["Visual similarity based on content analysis"]

    def find_similar_images(
        self, reference_image_id: str, limit: int = 5, prev_token: str = None
    ) -> ImageRecommendationResponse:
        """Find similar images based on a reference image"""
        # Get reference image
        reference_point = self.qdrant.retrieve(collection_name=self.collection_name, ids=[hash(reference_image_id)])[0]

        # Search for similar images
        search_results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=reference_point.vector,
            limit=limit + 1,  # +1 for next token
            offset=int(prev_token) if prev_token else 0,
        )

        # Process results
        recommendations = []
        next_token = None

        for i, result in enumerate(search_results):
            # Skip the reference image itself
            if result.id == hash(reference_image_id):
                continue

            if i < limit:
                metadata = ImageMetadata(**result.payload)
                similarity_aspects = self._calculate_similarity_aspects(reference_point.payload, result.payload)

                recommendations.append(
                    ImageRecommendation(
                        image=metadata, similarity_score=float(result.score), similarity_aspects=similarity_aspects
                    )
                )
            else:
                next_token = str(int(prev_token if prev_token else 0) + limit)
                break

        return ImageRecommendationResponse(recommendations=recommendations, next_token=next_token)
