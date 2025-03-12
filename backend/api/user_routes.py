from fastapi import APIRouter, HTTPException
from typing import List

from backend.models.user import UserQuery, UserMatchResponse
from backend.services.user_matching import UserMatchingService

router = APIRouter()
user_service = UserMatchingService()


@router.post("/match", response_model=UserMatchResponse)
async def find_matching_users(query: UserQuery) -> UserMatchResponse:
    """
    Find matching users based on natural language query.

    Args:
        query (UserQuery): Query parameters including search text and limit

    Returns:
        UserMatchResponse: List of matching users with compatibility scores
    """
    try:
        matches = user_service.find_matches(query.query, query.limit)
        return matches
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error finding matches: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint to verify API is running
    """
    return {"status": "healthy", "service": "user-matching"}
