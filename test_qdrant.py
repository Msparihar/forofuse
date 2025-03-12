import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import numpy as np

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def connect_to_qdrant():
    """Connect to local Qdrant instance"""
    try:
        client = QdrantClient("localhost", port=6333)
        print("Successfully connected to Qdrant")
        return client
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        return None


def get_image_embedding(image_path):
    """Get CLIP embedding for an image"""
    try:
        image = Image.open(image_path)
        inputs = processor(images=image, return_tensors="pt", padding=True)
        image_features = model.get_image_features(**inputs)
        # Normalize the features
        return image_features.detach().numpy()[0]
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None


def search_similar_images(client, query_vector, limit=5):
    """Search for similar images in Qdrant"""
    try:
        # Assuming collection name is 'midjourney_images'
        results = client.search(collection_name="midjourney-images", query_vector=query_vector, limit=limit)
        return results
    except Exception as e:
        print(f"Error searching similar images: {e}")
        return []


def main():
    # Connect to Qdrant
    client = connect_to_qdrant()
    if not client:
        print("Failed to connect to Qdrant. Exiting...")
        return

    # Test connection and get collection info
    try:
        collection_info = client.get_collection("midjourney-images")
        print(f"\nCollection info: {collection_info}")
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return

    # Example: Search using a test image
    test_image_path = "data/images/images.jpg"  # Test image path
    if os.path.exists(test_image_path):
        query_vector = get_image_embedding(test_image_path)
        if query_vector is not None:
            results = search_similar_images(client, query_vector)

            print("\nSearch Results:")
            for idx, result in enumerate(results, 1):
                print(f"\nResult {idx}:")
                print(f"Payload: {result.payload}")
                print(f"Score: {result.score}")
    else:
        print(f"Test image not found at {test_image_path}")


if __name__ == "__main__":
    main()
