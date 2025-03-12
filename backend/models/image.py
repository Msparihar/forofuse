from typing import List, Optional
from pydantic import BaseModel


class ImageMetadata(BaseModel):
    id: str
    filename: str
    labels: List[str]
    technical_metadata: dict
    content_features: dict


class ImageQuery(BaseModel):
    reference_image_id: str
    limit: Optional[int] = 5


class ImageRecommendation(BaseModel):
    image: ImageMetadata
    similarity_score: float
    similarity_aspects: List[str]


class ImageRecommendationResponse(BaseModel):
    recommendations: List[ImageRecommendation]
    next_token: Optional[str] = None  # For pagination/next functionality
