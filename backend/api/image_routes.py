from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Dict, Any
import os
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
import numpy as np

from PIL import Image

router = APIRouter()

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Qdrant client
qdrant_client = QdrantClient("localhost", port=6333)


def get_image_embedding(image_file) -> np.ndarray:
    """Generate embedding for an uploaded image"""
    try:
        image = Image.open(image_file)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**inputs)
        return image_features.detach().numpy()[0]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")


@router.post("/search")
async def search_similar_images(image: UploadFile = File(...), limit: int = 9) -> List[Dict[str, Any]]:
    """
    Find similar images in Midjourney dataset based on uploaded image

    Args:
        image: Image file to find similar images for
        limit: Maximum number of results to return (default: 9 for 3x3 grid)
    """
    try:
        # Generate embedding for uploaded image
        query_vector = get_image_embedding(image.file)

        # Search for similar images
        results = qdrant_client.search(collection_name="midjourney-images", query_vector=query_vector, limit=limit)

        # Format results
        similar_images = []
        for result in results:
            similar_images.append(
                {
                    "image_url": result.payload.get("image_url"),
                    "name": result.payload.get("name", "Unknown"),
                    "style": result.payload.get("url", "").replace("/styles/", "").replace("-", " ").title(),
                    "similarity_score": round((1 - float(result.score)) * 100, 2),  # Convert to percentage
                }
            )

        return similar_images
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding similar images: {str(e)}")


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "image-recommendation"}
