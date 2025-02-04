from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import logging
import os
import uuid
from datetime import datetime

from DocumentProcessor import DocumentProcessor
from PodcastGenerator import PodcastProcessor
from RagSystem import RAGSystem, process_files_with_progress
from models import ModelManager
from voice_config import VOICES

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

# Pydantic models
class ProcessRequest(BaseModel):
    file_paths: List[str]
    model_name: Optional[str] = "llama-3.3-70b-versatile"

class SessionResponse(BaseModel):
    session_id: str
    successful_documents: List[dict]
    failed_documents: List[dict]

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

class PodcastRequest(BaseModel):
    topic: str
    voice1: str
    voice2: str

class PodcastResponse(BaseModel):
    audio_url: str

class ModelSwitchRequest(BaseModel):
    model_name: str

class ModelSwitchResponse(BaseModel):
    message: str

# Create FastAPI app
app = FastAPI(
    title="Document Processing and RAG API",
    description="API for processing documents, generating podcasts, and answering queries using RAG",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active sessions
sessions: Dict[str, dict] = {}

@app.post("/api/session", response_model=SessionResponse)
async def create_session(request: ProcessRequest):
    """Create a new processing session and start document processing"""
    try:
        session_id = str(uuid.uuid4())
        processor = DocumentProcessor()
        rag = RAGSystem(request.model_name)
        
        # Process files
        results = await process_files_with_progress(processor, request.file_paths)
        
        # Separate successful and failed results
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Store successful results in RAG system
        if successful_results:
            await rag.ingest_documents(successful_results)
        
        # Store session data
        sessions[session_id] = {
            "rag": rag,
            "processor": processor,
            "successful_results": successful_results,
            "created_at": datetime.now()
        }
        
        return SessionResponse(
            session_id=session_id,
            successful_documents=[{
                "path": r.metadata.get("path"),
                "type": r.metadata.get("type")
            } for r in successful_results],
            failed_documents=[{
                "path": r.metadata.get("path"),
                "error": r.error_message
            } for r in failed_results]
        )
        
    except Exception as e:
        logging.error("Error creating session", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/{session_id}", response_model=QueryResponse)
async def query_documents(session_id: str, request: QueryRequest):
    """Query processed documents in a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        rag = sessions[session_id]["rag"]
        response = await rag.query(request.query)
        return QueryResponse(answer=response)
    
    except Exception as e:
        logging.error("Error processing query", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/podcast/{session_id}", response_model=PodcastResponse)
async def create_podcast(
    session_id: str,
    request: PodcastRequest,
    background_tasks: BackgroundTasks
):
    """Generate a podcast from processed documents"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        rag = sessions[session_id]["rag"]
        
        # Validate voice selection
        if request.voice1 not in VOICES or request.voice2 not in VOICES:
            raise HTTPException(status_code=400, detail="Invalid voice selection")
        
        # Initialize podcast processor
        podcast_processor = PodcastProcessor(
            user_id=os.getenv("PODCAST_USER_ID"),
            secret_key=os.getenv("PODCAST_SECRET_KEY")
        )
        
        # Get content from RAG
        content = await rag.query(request.topic)
        
        # Generate transcript
        transcript = await podcast_processor.process_podcast(content)
        
        # Generate audio in background
        audio_url = podcast_processor.generate_audio(
            transcript=transcript,
            host1_voice=request.voice1,
            host2_voice=request.voice2
        )
        
        return PodcastResponse(audio_url=audio_url)
        
    except Exception as e:
        logging.error("Error generating podcast", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/model/{session_id}", response_model=ModelSwitchResponse)
async def switch_model(session_id: str, request: ModelSwitchRequest):
    """Switch the language model for a session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions[session_id]
        
        # Create new RAG system with new model
        new_rag = RAGSystem(request.model_name)
        
        # Re-ingest documents with new model
        await new_rag.ingest_documents(session["successful_results"])
        
        # Update session
        session["rag"] = new_rag
        
        return ModelSwitchResponse(
            message=f"Successfully switched to model: {request.model_name}"
        )
        
    except Exception as e:
        logging.error("Error switching model", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models")
async def list_models():
    """List available language models"""
    try:
        model_manager = ModelManager()
        return model_manager.list_models()
    except Exception as e:
        logging.error("Error listing models", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/voices")
async def list_voices():
    """List available voices for podcast generation"""
    return {
        name: {
            "accent": info["accent"],
            "gender": info["gender"],
            "age": info["age"],
            "style": info["style"]
        }
        for name, info in VOICES.items()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)