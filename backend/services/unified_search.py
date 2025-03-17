from typing import Dict, List, Union
from PIL import Image
import torch
import logging
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedSearcher:
    def __init__(self, host: str = "localhost", port: int = 6333):
        """Initialize the unified search system with required models and client."""
        self.client = QdrantClient(host=host, port=port)
        self.setup_models()
        self.collection_name = "multimodal_collection"

    def setup_models(self):
        """Initialize CLIP and Sentence-Transformer models."""
        try:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise

    def search(self, query: Union[str, Image.Image], top_k: int = 5) -> Dict:
        """
        Unified search function that handles both text and image queries.

        Args:
            query: Either a string for text search or PIL Image for image search
            top_k: Number of results to return

        Returns:
            Dict containing search results and metadata
        """
        try:
            # Determine query type and process accordingly
            if isinstance(query, str):
                results = self._search_by_text(query, top_k)
            elif isinstance(query, Image.Image):
                results = self._search_by_image(query, top_k)
            else:
                raise ValueError("Query must be either text string or PIL Image")

            return self._format_results(results)

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return {"status": "error", "message": str(e), "results": []}

    def _search_by_text(self, query_text: str, top_k: int) -> List:
        """Process text search query."""
        # Process text and get embedding
        inputs = self.clip_processor(text=query_text, return_tensors="pt", padding=True, truncation=True, max_length=77)

        # Generate embedding
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
            query_vector = text_features.cpu().numpy().flatten()

        # Search in collection
        search_results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector.tolist(), limit=top_k
        )

        return search_results

    def _search_by_image(self, image: Image.Image, top_k: int) -> List:
        """Process image search query."""
        # Process image and get embedding
        inputs = self.clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(**inputs)
            query_vector = image_features.cpu().numpy().flatten()

        # Search in collection
        search_results = self.client.search(
            collection_name=self.collection_name, query_vector=query_vector.tolist(), limit=top_k
        )

        return search_results

    def _format_results(self, results: List) -> Dict:
        """Format search results into a standardized structure."""
        formatted_results = []

        for result in results:
            formatted_result = {
                "score": float(result.score),
                "product_info": {
                    "name": result.payload.get("Product Name", "Unknown"),
                    "category": result.payload.get("Category", "Unknown"),
                    "price": result.payload.get("Selling Price", "Unknown"),
                    "image_path": result.payload.get("image_path", "Unknown"),
                    "embedding_type": result.payload.get("embedding_type", "Unknown"),
                },
            }
            formatted_results.append(formatted_result)

        return {"status": "success", "count": len(formatted_results), "results": formatted_results}
