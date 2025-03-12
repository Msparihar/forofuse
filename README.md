# Forofuse

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

An innovative AI platform that combines a natural language user matching engine with a CLIP-based image recommendation system. Forofuse leverages modern AI techniques to enable discovery of like-minded individuals and to deliver image recommendations based on visual and contextual similarities.

## Features

### User Matching System

- Natural language queries to find like-minded people
- Vector-based similarity search using Qdrant
- Compatibility scoring and match explanations
- User profiles with interests, values, and expertise

### Image Recommendation System

- Content-based image similarity using CLIP model
- Multi-aspect similarity scoring
- Support for sequential recommendations
- Detailed similarity explanations

## Technical Stack

- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Vector Database**: Qdrant
- **ML Models**: CLIP for embeddings
- **Image Processing**: Pillow
- **Data Storage**: Local file system

## Prerequisites

- Python 3.9+
- Docker (for Qdrant)
- CUDA-compatible GPU (optional, for faster inference)

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd ai-matching-system
```

2. Install dependencies:

```bash
uv pip install -r requirements.txt
```

3. Start Qdrant using Docker:

```bash
docker run -d --name qdrant -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

## Running the Application

1. Start the application using the provided script:

```bash
python run_app.py
```

This will:

- Start Qdrant if not running
- Launch the FastAPI backend
- Start the Streamlit frontend
- Open the application in your default browser

The application will be available at:

- Frontend: <http://localhost:8501>
- Backend API: <http://localhost:8000>
- API Documentation: <http://localhost:8000/docs>

## Usage

### User Matching

1. Navigate to the User Matching page
2. Enter a natural language query describing the kind of people you're looking for
3. Adjust the number of matches you want to see
4. View detailed profiles with compatibility scores and match reasons

Example query:

```
I'm looking for AI researchers in NYC who are passionate about environmental conservation and hiking
```

### Image Recommendations

1. Navigate to the Image Recommendation page
2. Upload images with descriptive labels
3. Select a reference image
4. View similar images with similarity scores and explanations
5. Use the "Next" feature to explore more recommendations

## Project Structure

```
.
├── backend/
│   ├── api/
│   │   ├── main.py
│   │   ├── user_routes.py
│   │   └── image_routes.py
│   ├── models/
│   │   ├── user.py
│   │   └── image.py
│   └── services/
│       ├── user_matching.py
│       └── image_recommendation.py
├── frontend/
│   └── pages/
│       ├── user_matching.py
│       └── image_recommendation.py
├── data/
│   ├── users.json
│   └── images/
├── requirements.txt
└── run_app.py
```

## API Documentation

### User Matching Endpoints

- `POST /api/users/match`: Find matching users
  - Parameters: query text, limit
  - Returns: Ranked list of matches with compatibility scores

### Image Recommendation Endpoints

- `POST /api/images/upload`: Upload and index a new image
  - Parameters: image file, labels
  - Returns: Image ID and confirmation

- `POST /api/images/recommend`: Get similar images
  - Parameters: reference image ID, limit
  - Returns: Ranked list of similar images with scores

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - feel free to use this project for any purpose.
