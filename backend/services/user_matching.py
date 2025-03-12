import json
from typing import List, Dict, Any
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from transformers import CLIPTextModel, CLIPTokenizer
import torch
import numpy as np

from backend.models.user import User, UserMatch, UserMatchResponse


class UserMatchingService:
    def __init__(self):
        # Initialize CLIP model for text embeddings
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize Qdrant client
        self.qdrant = QdrantClient("localhost", port=6333)
        self.collection_name = "user_profiles"

        # Create collection if it doesn't exist
        self._initialize_collection()

        # Load and index users if collection is empty
        self._load_initial_users()

    def _initialize_collection(self):
        """Initialize Qdrant collection for user profiles"""
        try:
            self.qdrant.get_collection(self.collection_name)
        except Exception:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=512,  # CLIP embedding size
                    distance=models.Distance.COSINE,
                ),
            )

    def _load_initial_users(self):
        """Load initial users from JSON file and index them"""
        try:
            collections_info = self.qdrant.get_collection(self.collection_name)
            if collections_info.points_count == 0:
                users_file = Path("data/users.json")
                if users_file.exists():
                    with open(users_file, "r") as f:
                        users_data = json.load(f)
                        for user in users_data["users"]:
                            self.index_user(User(**user))
        except Exception as e:
            print(f"Error loading initial users: {e}")

    def _generate_user_embedding(self, user: User) -> np.ndarray:
        """Generate embedding for a user profile using CLIP"""
        # Combine relevant user information into a text description
        user_text = f"{user.basic_info.profession} interested in {', '.join(user.interests)}. "
        user_text += f"Values include {', '.join(user.values)}. "
        user_text += f"Expert in {', '.join(user.expertise.areas)}. "
        user_text += f"Prefers {user.preferences.collaboration_style} collaboration style."

        # Generate embedding using CLIP
        inputs = self.tokenizer(user_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)

        return embedding.numpy()[0]

    def _generate_query_embedding(self, query: str) -> np.ndarray:
        """Generate embedding for a search query using CLIP"""
        inputs = self.tokenizer(query, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embedding = self.model(**inputs).last_hidden_state.mean(dim=1)
        return embedding.numpy()[0]

    def index_user(self, user: User):
        """Index a user profile in Qdrant"""
        embedding = self._generate_user_embedding(user)

        self.qdrant.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=hash(user.id),  # Use hash of user ID as point ID
                    vector=embedding.tolist(),
                    payload=json.loads(user.model_dump_json()),
                )
            ],
        )

    def _calculate_match_reasons(self, query: str, user: User) -> List[str]:
        """Generate reasons for why a user matches the query"""
        reasons = []

        # Extract key terms from query (simplified version)
        query_terms = set(query.lower().split())

        # Check interests match
        matching_interests = [
            interest for interest in user.interests if any(term in interest.lower() for term in query_terms)
        ]
        if matching_interests:
            reasons.append(f"Shares interests in: {', '.join(matching_interests)}")

        # Check values match
        matching_values = [value for value in user.values if any(term in value.lower() for term in query_terms)]
        if matching_values:
            reasons.append(f"Aligned values: {', '.join(matching_values)}")

        # Check expertise match
        matching_expertise = [
            area for area in user.expertise.areas if any(term in area.lower() for term in query_terms)
        ]
        if matching_expertise:
            reasons.append(f"Expert in: {', '.join(matching_expertise)}")

        return reasons or ["Profile aligns with search criteria"]

    def find_matches(self, query: str, limit: int = 5) -> UserMatchResponse:
        """Find matching users based on a natural language query"""
        # Generate query embedding
        query_embedding = self._generate_query_embedding(query)

        # Search in Qdrant
        search_results = self.qdrant.search(
            collection_name=self.collection_name, query_vector=query_embedding.tolist(), limit=limit
        )

        # Process results
        matches = []
        for result in search_results:
            user = User(**result.payload)
            match_reasons = self._calculate_match_reasons(query, user)
            matches.append(UserMatch(user=user, compatibility_score=float(result.score), match_reasons=match_reasons))

        return UserMatchResponse(matches=matches, query_understanding=f"Looking for users matching: {query}")
