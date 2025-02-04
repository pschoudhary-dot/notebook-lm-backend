import streamlit as st
import requests
import os
from voice_config import VOICES
import tempfile
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ui.log'),
        logging.StreamHandler()
    ]
)

# Configuration
API_BASE_URL = "http://127.0.0.1:8000/api"
TEMP_DIR = tempfile.mkdtemp()  # Create a temporary directory

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "processed" not in st.session_state:
    st.session_state.processed = False
if "current_model" not in st.session_state:
    st.session_state.current_model = None


def get_models():
    """Fetch available models from the API"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch models: {str(e)}")
        return []


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary directory"""
    try:
        if uploaded_file is None:
            return None
        file_path = os.path.join(TEMP_DIR, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        logging.error(f"Error saving uploaded file: {str(e)}")
        return None


def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        for file in os.listdir(TEMP_DIR):
            file_path = os.path.join(TEMP_DIR, file)
            try:
                os.remove(file_path)
            except Exception as e:
                logging.error(f"Error removing file {file_path}: {str(e)}")
    except Exception as e:
        logging.error(f"Error cleaning up temp files: {str(e)}")


# Page config
st.set_page_config(
    page_title="Document Processor & Podcast Generator",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("📚 Document Processor & Podcast Generator")
st.markdown("""
### AI-powered document analysis and podcast creation
Process documents, ask questions, and create AI-powered podcasts from your content.
""")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    # Model selection
    models = get_models()
    if models:
        default_model = "llama-3.3-70b-versatile"
        selected_model = st.selectbox(
            "Select Language Model",
            options=[m["name"] for m in models],
            index=[m["name"] for m in models].index(default_model) if default_model in [m["name"] for m in models] else 0
        )
        # Display model details
        selected_model_info = next((m for m in models if m["name"] == selected_model), None)
        if selected_model_info:
            st.markdown("**Model Details:**")
            st.markdown(f"""
            Provider: {selected_model_info['provider']}
            Cost: ${selected_model_info['cost_per_million_tokens']}/M tokens
            """)
    else:
        st.error("⚠️ No models available")
        selected_model = None

    # Voice selection section
    st.markdown("---")
    st.markdown("🎙️ **Available Voices**")
    for voice, details in VOICES.items():
        with st.expander(voice):
            st.write(f"🌍 Accent: {details['accent']}")
            st.write(f"⚧ Gender: {details['gender']}")
            st.write(f"📅 Age: {details['age']}")
            st.write(f"🎭 Style: {details['style']}")

# Main content
st.header("1. Document Input")

# Document input section
col1, col2 = st.columns(2)
with col1:
    st.subheader("URLs")
    url_input = st.text_area(
        "Enter document URLs (one per line)",
        placeholder="https://example.com/doc1\nhttps://example.com/doc2",
        help="Enter URLs of web pages or documents you want to process"
    )

with col2:
    st.subheader("File Upload")
    uploaded_files = st.file_uploader(
        "Upload local files",
        accept_multiple_files=True,
        type=["txt", "pdf", "docx", "mp3", "mp4", "wav"],
        help="Upload documents from your computer"
    )

# Process files
file_paths = []

# Add URLs to file paths
if url_input:
    file_paths.extend([url.strip() for url in url_input.split("\n") if url.strip()])

# Add uploaded files to file paths
if uploaded_files:
    for uploaded_file in uploaded_files:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            file_paths.append(file_path)

# Process button
st.markdown("---")
col1, col2 = st.columns([2, 1])
with col1:
    process_button = st.button(
        "🚀 Process Documents",
        disabled=len(file_paths) == 0,
        help="Start processing the provided documents",
        use_container_width=True
    )

with col2:
    if st.session_state.processed:
        if st.button("🗑️ Clear Session", use_container_width=True):
            try:
                if st.session_state.session_id:
                    requests.delete(f"{API_BASE_URL}/session/{st.session_state.session_id}")
                    st.session_state.session_id = None
                    st.session_state.processed = False
                    cleanup_temp_files()
                    st.experimental_rerun()
            except Exception as e:
                st.error(f"Failed to clear session: {str(e)}")

if process_button:
    with st.spinner("🔄 Processing documents..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/session",
                json={
                    "file_paths": file_paths,
                    "model_name": selected_model
                }
            )
            response.raise_for_status()
            result = response.json()
            st.session_state.session_id = result["session_id"]
            st.session_state.processed = True
            st.session_state.current_model = selected_model
            st.success("✅ Processing complete!")

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**📄 Successfully Processed:**")
                for doc in result["successful_documents"]:
                    st.success(f"✓ {os.path.basename(doc.get('path', 'Unknown'))}")
            with col2:
                if result["failed_documents"]:
                    st.markdown("**❌ Failed to Process:**")
                    for doc in result["failed_documents"]:
                        st.error(f"✗ {os.path.basename(doc['path'])}: {doc['error']}")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Failed to connect to the API server. Make sure it's running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"⚠️ Processing failed: {str(e)}")

# Interactive section (only shown after processing)
if st.session_state.processed and st.session_state.session_id:
    st.markdown("---")
    st.header("2. Document Interaction")
    tabs = st.tabs(["💭 Ask Questions", "🎙️ Create Podcast", "ℹ️ Session Info"])

    # Questions tab
    with tabs[0]:
        # Model switching
        if selected_model != st.session_state.current_model:
            if st.button("🔄 Switch Model"):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/model/{st.session_state.session_id}",
                        json={"model_name": selected_model}
                    )
                    response.raise_for_status()
                    st.session_state.current_model = selected_model
                    st.success(f"✅ Switched to {selected_model}")
                except Exception as e:
                    st.error(f"⚠️ Model switch failed: {str(e)}")

        query = st.text_area("Ask a question about your documents", placeholder="What are the main points discussed in the documents?")
        if query:
            with st.spinner("🔍 Searching documents..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/query/{st.session_state.session_id}",
                        json={"query": query}
                    )
                    response.raise_for_status()
                    answer = response.json()["answer"]
                    st.markdown("### 💡 Answer")
                    st.markdown(answer)
                except Exception as e:
                    st.error(f"⚠️ Query failed: {str(e)}")

    # Podcast tab
    with tabs[1]:
        st.markdown("### 🎙️ Create a Podcast")
        podcast_topic = st.text_area(
            "What should the podcast be about?",
            placeholder="Describe the topic or provide specific questions to discuss"
        )
        col1, col2 = st.columns(2)
        with col1:
            voice1 = st.selectbox("Select Voice for Host 1", options=list(VOICES.keys()))
        with col2:
            voice2 = st.selectbox("Select Voice for Host 2", options=list(VOICES.keys()))

        if st.button("🎧 Generate Podcast", disabled=not podcast_topic):
            with st.spinner("🎵 Generating podcast... This may take a few minutes."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/podcast/{st.session_state.session_id}",
                        json={
                            "topic": podcast_topic,
                            "voice1": voice1,
                            "voice2": voice2
                        }
                    )
                    response.raise_for_status()
                    result = response.json()
                    audio_url = result["audio_url"]
                    st.success("✅ Podcast generated successfully!")
                    st.audio(audio_url)
                    # Download button
                    st.markdown(f"📥 [Download Podcast]({audio_url})")
                except Exception as e:
                    st.error(f"⚠️ Podcast generation failed: {str(e)}")

    # Session info tab
    with tabs[2]:
        try:
            response = requests.get(
                f"{API_BASE_URL}/session/{st.session_state.session_id}/info"
            )
            response.raise_for_status()
            info = response.json()
            st.markdown("### 📊 Session Information")
            st.json(info)
        except Exception as e:
            st.error(f"⚠️ Failed to get session info: {str(e)}")

# Footer
st.markdown("---")
footer_col1, footer_col2 = st.columns(2)
with footer_col1:
    st.markdown("ℹ️ Make sure the API server is running before using this app")
with footer_col2:
    st.markdown("🔧 API Status: " + ("🟢 Connected" if get_models() else "🔴 Not Connected"))