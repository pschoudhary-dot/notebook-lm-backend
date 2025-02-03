import streamlit as st
import requests
import uuid
import os
from voice_config import VOICES

# Configuration
API_BASE_URL = "http://127.0.0.1:8000"  # Update this if your API is hosted elsewhere
TEMP_DIR = "./temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "processed" not in st.session_state:
    st.session_state.processed = False

def get_models():
    response = requests.get(f"{API_BASE_URL}/models")
    return response.json()["models"]

def save_uploaded_file(uploaded_file):
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

st.title("📚 Document Processor & Podcast Generator")
st.markdown("### AI-powered document analysis and podcast creation")

# Sidebar for settings and info
with st.sidebar:
    st.header("Settings")
    models = get_models()
    default_model = next((m for m in models if m["name"] == "llama-3.3-70b-versatile"), models[0])
    selected_model = st.selectbox(
        "Select Model",
        options=[m["name"] for m in models],
        index=[m["name"] for m in models].index(default_model["name"])
    )
    
    st.markdown("---")
    st.markdown("**Available Voices**")
    st.write(VOICES)

# Main processing section
st.header("1. Process Documents")
file_input = st.text_input("Enter document URLs (comma-separated)", 
                          value="https://genius.com/Lord-huron-mine-forever-lyrics")
uploaded_files = st.file_uploader("Or upload files", accept_multiple_files=True, 
                                 type=["txt", "pdf", "docx", "mp3", "mp4"])

file_paths = []
if file_input:
    file_paths.extend([url.strip() for url in file_input.split(",") if url.strip()])
if uploaded_files:
    file_paths.extend([save_uploaded_file(f) for f in uploaded_files])

if st.button("Process Documents") and file_paths:
    with st.spinner("Processing documents..."):
        try:
            response = requests.post(
                f"{API_BASE_URL}/sessions",
                json={
                    "file_paths": file_paths,
                    "model_name": selected_model
                }
            )
            response.raise_for_status()
            
            result = response.json()
            st.session_state.session_id = result["session_id"]
            st.session_state.processed = True
            
            st.success("Processing complete!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Successful Documents:** {len(result['successful_documents'])}")
                for doc in result["successful_documents"]:
                    st.write(f"- {doc.get('path', 'Unknown')}")
            
            with col2:
                if result["failed_documents"]:
                    st.markdown("**Failed Documents**")
                    for doc in result["failed_documents"]:
                        st.error(f"- {doc['path']}: {doc['error']}")
            
        except Exception as e:
            st.error(f"Processing failed: {str(e)}")

# Chat and podcast section
if st.session_state.processed and st.session_state.session_id:
    st.header("2. Document Interaction")
    
    tab1, tab2, tab3 = st.tabs(["Ask Questions", "Create Podcast", "Session Info"])
    
    with tab1:
        query = st.text_input("Ask a question about your documents")
        if query:
            with st.spinner("Searching documents..."):
                try:
                    response = requests.post(
                        f"{API_BASE_URL}/sessions/{st.session_state.session_id}/query",
                        json={"query": query}
                    )
                    response.raise_for_status()
                    
                    answer = response.json()["answer"]
                    st.markdown(f"**Answer:** {answer}")
                    
                except Exception as e:
                    st.error(f"Query failed: {str(e)}")
        
        if st.button("Switch Model"):
            try:
                response = requests.put(
                    f"{API_BASE_URL}/sessions/{st.session_state.session_id}/model",
                    json={"model_name": selected_model}
                )
                response.raise_for_status()
                st.success(f"Switched to {selected_model} model")
            except Exception as e:
                st.error(f"Model switch failed: {str(e)}")
    
    with tab2:
        podcast_topic = st.text_input("Podcast Topic")
        col1, col2 = st.columns(2)
        with col1:
            voice1 = st.selectbox("Host 1 Voice", options=VOICES)
        with col2:
            voice2 = st.selectbox("Host 2 Voice", options=VOICES)
        
        if st.button("Generate Podcast"):
            if not podcast_topic:
                st.warning("Please enter a podcast topic")
            else:
                with st.spinner("Generating podcast..."):
                    try:
                        response = requests.post(
                            f"{API_BASE_URL}/sessions/{st.session_state.session_id}/podcast",
                            json={
                                "topic": podcast_topic,
                                "voice1": voice1,
                                "voice2": voice2
                            }
                        )
                        response.raise_for_status()
                        
                        audio_url = response.json()["audio_url"]
                        st.audio(audio_url)
                        st.success("Podcast generated successfully!")
                        
                    except Exception as e:
                        st.error(f"Podcast generation failed: {str(e)}")
    
    with tab3:
        try:
            response = requests.get(
                f"{API_BASE_URL}/sessions/{st.session_state.session_id}/info"
            )
            response.raise_for_status()
            
            info = response.json()
            st.json(info)
            
        except Exception as e:
            st.error(f"Failed to get session info: {str(e)}")

# Cleanup
if st.session_state.session_id and st.button("Clear Session"):
    requests.delete(f"{API_BASE_URL}/sessions/{st.session_state.session_id}")
    st.session_state.session_id = None
    st.session_state.processed = False
    st.experimental_rerun()

st.markdown("---")
st.markdown("ℹ️ Note: Make sure the API server is running before using this app")