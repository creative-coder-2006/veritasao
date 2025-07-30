# app.py

"""
VERITAS.AI Streamlit Application
================================

This is the main entry point and user interface for the VERITAS.AI application.
It provides a web-based dashboard for users to analyze content from various
platforms (News, Reddit, YouTube).

Key Features:
-   Interactive UI built with Streamlit.
-   User authentication (login/signup).
-   Platform selection and URL input.
-   Orchestration of scraping and analysis pipelines.
-   Dynamic display of analysis scores and explanations.
-   Historical analysis tracking for logged-in users.
-   Robust error handling and user feedback.
"""

import streamlit as st
from scraper import NewsScraper, RedditScraper, VideoScraper
from analyzer import AnalysisPipeline
from auth import Authenticator
from database import (
    init_db, add_analysis_record, get_user_history,
    flag_source, get_source_flags
)
from utils import get_domain_from_url
import logging
import json

# --- Page Configuration ---
st.set_page_config(
    page_title="VERITAS.AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Initialize Modules ---
# Use session state to initialize these once to save resources
if 'authenticator' not in st.session_state:
    st.session_state.authenticator = Authenticator()
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = AnalysisPipeline()
if 'scrapers' not in st.session_state:
    st.session_state.scrapers = {
        "News": NewsScraper(),
        "Reddit": RedditScraper(),
        "YouTube": VideoScraper()
    }

# Initialize the database
init_db()


# --- UI Helper Functions ---
def display_login_form():
    """Displays the login and registration forms in the sidebar."""
    st.sidebar.title("Login / Register")
    form_choice = st.sidebar.radio("Choose Action", ["Login", "Register"])

    with st.sidebar.form(key=form_choice.lower()):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_button = st.form_submit_button(label=form_choice)

        if submit_button:
            if form_choice == "Login":
                if st.session_state.authenticator.login(username, password):
                    st.session_state.logged_in = True
                    st.experimental_rerun()
                else:
                    st.sidebar.error("Invalid username or password.")
            elif form_choice == "Register":
                if st.session_state.authenticator.register_user(username, password):
                    st.sidebar.success("Registration successful! Please log in.")
                else:
                    st.sidebar.error("Username already exists or invalid input.")

def display_results(results: dict):
    """Renders the analysis results in a structured format."""
    st.header("Analysis Report")

    if "error" in results:
        st.error(f"Analysis failed: {results['error']}")
        return

    scores = results.get("scores", {})
    explanation = results.get("explanation", {})
    metadata = results.get("metadata", {})

    # --- Main Scores ---
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Trust Score",
            f"{scores.get('trust_score', 0):.2f}/1.00",
            help="Overall trustworthiness. Higher is better."
        )
    with col2:
        st.metric(
            "Misinformation",
            f"{scores.get('misinformation_score', 0):.0%}",
            help="Probability the content contains misinformation. Lower is better."
        )
    with col3:
        st.metric(
            "Credibility Score",
            f"{scores.get('credibility_score', 0):.2f}/1.00",
            help="Source and content quality. Higher is better."
        )

    # --- Explanation ---
    st.subheader("AI-Generated Explanation")
    st.markdown(explanation.get("summary", "No explanation available."))
    if explanation.get("status") == "fallback":
        st.warning("This is a basic summary. Configure the OpenAI API key for advanced explanations.")

    # --- Detailed Breakdown ---
    with st.expander("Show Detailed Analysis Breakdown"):
        st.subheader("Risk Factors")
        p1, p2 = st.columns(2)
        p1.progress(scores.get('llm_origin_probability', 0), text=f"AI-Generated Text Probability")
        if results['platform'] == 'youtube':
            p2.progress(scores.get('deepfake_probability', 0), text=f"Deepfake Video Probability")

        st.subheader("Raw Analysis Data")
        st.json(results.get("detailed_analysis", {}))

    # --- Source Information & Flagging ---
    st.sidebar.subheader("Source Information")
    domain = get_domain_from_url(results.get("url", ""))
    st.sidebar.write(f"**Source Domain:** `{domain}`")
    
    flag_count = get_source_flags(domain)
    st.sidebar.warning(f"This source has been flagged **{flag_count}** time(s) by users.")

    if st.sidebar.button("🚩 Flag this Source as Unreliable"):
        if flag_source(domain, st.session_state.authenticator.get_current_user()):
            st.sidebar.success(f"Successfully flagged '{domain}'.")
            st.experimental_rerun()
        else:
            st.sidebar.error("You have already flagged this source.")


# --- Main Application Logic ---
def main():
    """The main function that runs the Streamlit app."""
    st.title("🤖 VERITAS.AI")
    st.markdown("---")

    # --- Authentication Check ---
    if not st.session_state.get("logged_in", False):
        display_login_form()
        st.info("Please log in or register to use the application.")
        return

    # --- Logged-in View ---
    st.sidebar.success(f"Logged in as **{st.session_state.authenticator.get_current_user()}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticator.logout()
        st.session_state.logged_in = False
        st.experimental_rerun()
    
    st.sidebar.markdown("---")
    
    # --- Main Interface ---
    st.header("Analyze New Content")
    platform = st.selectbox("Select Platform", ["News", "Reddit", "YouTube"])
    url = st.text_input("Enter URL to Analyze", "")

    if st.button("Analyze"):
        if not url:
            st.warning("Please enter a URL.")
            return

        with st.spinner(f"Analyzing content from {url}... This may take a moment."):
            try:
                # 1. Scrape
                scraper = st.session_state.scrapers[platform]
                scraped_data = scraper.scrape(url)
                if "error" in scraped_data:
                    st.error(f"Scraping failed: {scraped_data['error']}")
                    return

                # 2. Analyze
                pipeline = st.session_state.pipeline
                results = pipeline.run(scraped_data)

                # 3. Save & Display
                if "error" not in results:
                    add_analysis_record(
                        user_id=st.session_state.authenticator.get_current_user_id(),
                        url=url,
                        platform=platform,
                        scores=json.dumps(results.get("scores", {})),
                        explanation=results.get("explanation", {}).get("summary", "")
                    )
                    st.session_state.last_analysis = results
                else:
                    st.error(f"Analysis pipeline failed: {results.get('error')}")

            except Exception as e:
                logger.error(f"An unexpected error occurred during analysis: {e}")
                st.error("A critical error occurred. Please check the logs or try again.")
                st.exception(e)

    # Display the results of the last analysis
    if "last_analysis" in st.session_state:
        display_results(st.session_state.last_analysis)
        st.markdown("---")

    # --- History View ---
    st.header("Your Analysis History")
    history = get_user_history(st.session_state.authenticator.get_current_user_id())
    if history:
        for record in history:
            col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
            col1.write(record['url'])
            col2.info(record['platform'])
            # Safely parse scores
            try:
                scores_data = json.loads(record['scores'])
                col3.metric("Trust Score", f"{scores_data.get('trust_score', 0):.2f}")
            except (json.JSONDecodeError, TypeError):
                col3.write("N/A")
            col4.write(record['timestamp'])
    else:
        st.info("You have no analysis history yet.")


if __name__ == "__main__":
    main()