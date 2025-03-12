import streamlit as st
import requests
from PIL import Image
import io

# Configure page
st.set_page_config(page_title="Image Similarity Search", page_icon="🔍", layout="wide")

# Constants
API_URL = "http://localhost:8000/api/images"
THUMBNAIL_WIDTH = 300


def search_similar_images(image_file):
    """Search for similar images using the API"""
    try:
        files = {"image": image_file}
        response = requests.post(f"{API_URL}/search", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error searching similar images: {str(e)}")
        return None


def display_image_grid(results):
    """Display images in a 3x3 grid with metadata"""
    if not results:
        return

    # Create rows with 3 columns each
    for i in range(0, len(results), 3):
        row = results[i : i + 3]
        cols = st.columns(3)

        for col, result in zip(cols, row):
            with col:
                try:
                    # Load and display image
                    response = requests.get(result["image_url"])
                    img = Image.open(io.BytesIO(response.content))
                    st.image(img, width=THUMBNAIL_WIDTH)

                    # Display metadata
                    st.markdown(f"**{result['name']}**")
                    st.markdown(f"Style: {result['style']}")
                    st.metric("Similarity", f"{result['similarity_score']}%", delta=None, delta_color="normal")
                except Exception as e:
                    st.error(f"Error loading image: {str(e)}")


def main():
    st.title("🔍 Image Similarity Search")
    st.write("""
    Upload an image to find visually similar images from the Midjourney dataset.
    The system uses CLIP model to understand visual features and find matches.
    """)

    # Image upload
    uploaded_file = st.file_uploader(
        "Choose an image to find similar ones",
        type=["png", "jpg", "jpeg"],
        help="Upload an image to search for visually similar images",
    )

    if uploaded_file:
        # Show uploaded image
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Your Image")
            st.image(uploaded_file, width=THUMBNAIL_WIDTH)

        # Search for similar images
        with st.spinner("Searching for similar images..."):
            results = search_similar_images(uploaded_file)

            if results:
                st.subheader("Similar Images")
                display_image_grid(results)


if __name__ == "__main__":
    main()
