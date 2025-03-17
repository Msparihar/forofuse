import streamlit as st
import requests
from PIL import Image
import io
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parents[2])
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.services.unified_search import UnifiedSearcher

st.set_page_config(page_title="Product Search", page_icon="🔍", layout="wide")


# Initialize the searcher
@st.cache_resource
def get_searcher():
    return UnifiedSearcher()


def load_image(image_file):
    """Load and return a PIL Image from uploaded file."""
    return Image.open(image_file)


def display_results(results):
    """Display search results in a grid layout."""
    if results["status"] == "error":
        st.error(f"Search error: {results['message']}")
        return

    if not results["results"]:
        st.info("No results found.")
        return

    # Create columns for grid layout
    cols = st.columns(3)

    for idx, result in enumerate(results["results"]):
        with cols[idx % 3]:
            product = result["product_info"]

            # Display product image if available
            if os.path.exists(product["image_path"]):
                st.image(product["image_path"], caption=product["name"])

            # Display product information
            st.write(f"**{product['name']}**")
            st.write(f"Category: {product['category']}")
            st.write(f"Price: ${product['price']}")
            st.write(f"Similarity Score: {result['score']:.4f}")
            st.divider()


def main():
    st.title("Product Search")
    st.write("Search products using text or image")

    # Initialize searcher
    searcher = get_searcher()

    # Create tabs for different search methods
    tab1, tab2 = st.tabs(["Text Search", "Image Search"])

    with tab1:
        # Text search interface
        query_text = st.text_input("Enter your search query")
        if st.button("Search", key="text_search"):
            if query_text:
                with st.spinner("Searching..."):
                    results = searcher.search(query_text)
                    display_results(results)
            else:
                st.warning("Please enter a search query")

    with tab2:
        # Image search interface
        uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = load_image(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Search Similar Products", key="image_search"):
                with st.spinner("Searching..."):
                    results = searcher.search(image)
                    display_results(results)

    # Add filters
    with st.sidebar:
        st.header("Filters")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        st.info("""
        Search Tips:
        - For text search, try describing the product features
        - For image search, upload a clear product image
        - Adjust the number of results using the slider
        """)


if __name__ == "__main__":
    main()
