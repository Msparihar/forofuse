import streamlit as st
import requests
import json
from typing import Dict, Any

# Configure page
st.set_page_config(page_title="User Matching System", page_icon="👥", layout="wide")

# Constants
API_URL = "http://localhost:8000/api"


def search_users(query: str, limit: int = 5) -> Dict[str, Any]:
    """
    Send search query to backend API and return matches
    """
    try:
        response = requests.post(f"{API_URL}/users/match", json={"query": query, "limit": limit})
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {str(e)}")
        return None


def display_user_card(user: Dict[str, Any], compatibility: float, reasons: list):
    """
    Display a user profile card with match information
    """
    with st.container():
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader(f"{user['basic_info']['name']} ({user['basic_info']['age']})")
            st.write(f"📍 {user['basic_info']['location']}")
            st.write(f"💼 {user['basic_info']['profession']}")

            # Display interests
            st.write("**Interests:**")
            st.write(", ".join(user["interests"]))

            # Display values
            st.write("**Values:**")
            st.write(", ".join(user["values"]))

            # Display expertise
            st.write("**Expertise:**")
            st.write(f"Level: {user['expertise']['level']}")
            st.write("Areas: " + ", ".join(user["expertise"]["areas"]))

        with col2:
            # Display compatibility score
            st.metric(label="Compatibility", value=f"{compatibility * 100:.1f}%")

            # Display match reasons
            st.write("**Why this match?**")
            for reason in reasons:
                st.write(f"• {reason}")

        st.divider()


def main():
    st.title("👥 User Matching System")
    st.write("""
    Find like-minded people based on your interests, values, and preferences.
    Use natural language to describe what you're looking for!
    """)

    # Search interface
    with st.form("search_form"):
        query = st.text_area(
            "What kind of people are you looking to connect with?",
            placeholder="Example: I'm looking for AI researchers in NYC who are passionate about environmental conservation and hiking",
            height=100,
        )

        col1, col2 = st.columns([3, 1])
        with col2:
            limit = st.number_input("Number of matches", min_value=1, max_value=10, value=5)

        submitted = st.form_submit_button("🔍 Find Matches")

    # Process search when form is submitted
    if submitted and query:
        with st.spinner("Finding matches..."):
            results = search_users(query, limit)

            if results:
                st.write("### 🎯 Matching Results")
                st.write(f"*{results['query_understanding']}*")

                for match in results["matches"]:
                    display_user_card(
                        user=match["user"], compatibility=match["compatibility_score"], reasons=match["match_reasons"]
                    )
            else:
                st.warning("No matches found. Try adjusting your search criteria.")


if __name__ == "__main__":
    main()
