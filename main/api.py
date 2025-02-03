from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import uuid
import asyncio
from DocumentProcessor import DocumentProcessor
from PodcastGenerator import PodcastProcessor
from RagSystem import RAGSystem, process_files_with_progress
from models import ModelManager
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Pydantic models for request/response validation
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

# Session management
class Session:
    def __init__(self, rag_system, successful_docs, processor):
        self.rag = rag_system
        self.successful_docs = successful_docs
        self.processor = processor

class SessionManager:
    def __init__(self):
        self.sessions = {}
        self.podcast_processor = PodcastProcessor(
            user_id=os.getenv("PODCAST_USER_ID", "ssFAgjpDO5NVuWyC9AOLghsnTlS2"),
            secret_key=os.getenv("PODCAST_SECRET_KEY", "ak-f16737b738ea4dd78ea3aefa81fb9c25")
        )

    async def create_session(self, file_paths, model_name=None):
        try:
            processor = DocumentProcessor()
            model_manager = ModelManager()
            
            if model_name and not model_manager.model_exists(model_name):
                raise ValueError(f"Model {model_name} not found")
            
            model_name = model_name or "llama-3.3-70b-versatile"
            rag = RAGSystem(model_name)
            
            results = await process_files_with_progress(processor, file_paths)
            successful = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            
            if successful:
                await rag.ingest_documents(successful)
            
            session_id = str(uuid.uuid4())
            self.sessions[session_id] = Session(rag, successful, processor)
            return session_id, successful, failed
        
        except Exception as e:
            logging.error(f"Session creation failed: {str(e)}")
            raise

    def get_session(self, session_id):
        return self.sessions.get(session_id)

    def delete_session(self, session_id):
        if session_id in self.sessions:
            try:
                # Cleanup temporary files
                session = self.sessions[session_id]
                for file in os.listdir(session.processor.tmp_dir):
                    os.remove(os.path.join(session.processor.tmp_dir, file))
                os.rmdir(session.processor.tmp_dir)
            except Exception as e:
                logging.warning(f"Cleanup error: {str(e)}")
            del self.sessions[session_id]

manager = SessionManager()

# API endpoints
@app.post("/sessions", response_model=SessionResponse)
async def create_session(request: ProcessRequest):
    try:
        session_id, successful, failed = await manager.create_session(
            request.file_paths, request.model_name
        )
        return SessionResponse(
            session_id=session_id,
            successful_documents=[doc.metadata for doc in successful],
            failed_documents=[{
                "path": doc.metadata.get("path", "unknown"), 
                "error": doc.error_message
            } for doc in failed]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/sessions/{session_id}/query", response_model=QueryResponse)
async def query_session(session_id: str, request: QueryRequest):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        response = await session.rag.query(request.query)
        return QueryResponse(answer=response)
    except Exception as e:
        logging.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/podcast", response_model=PodcastResponse)
async def create_podcast(session_id: str, request: PodcastRequest):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.successful_docs:
        raise HTTPException(status_code=400, detail="No documents available for podcast")
    
    try:
        podcast_content = await session.rag.query(request.topic)
        transcript = await manager.podcast_processor.process_podcast(podcast_content)
        audio_url = manager.podcast_processor.generate_audio(
            transcript=transcript,
            host1_voice=request.voice1,
            host2_voice=request.voice2
        )
        return PodcastResponse(audio_url=audio_url)
    except Exception as e:
        logging.error(f"Podcast generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/sessions/{session_id}/model", response_model=ModelSwitchResponse)
async def switch_model(session_id: str, request: ModelSwitchRequest):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    model_manager = ModelManager()
    if not model_manager.model_exists(request.model_name):
        raise HTTPException(status_code=400, detail=f"Model {request.model_name} not found")
    
    try:
        new_rag = RAGSystem(request.model_name)
        await new_rag.ingest_documents(session.successful_docs)
        session.rag = new_rag
        return ModelSwitchResponse(message=f"Switched to model {request.model_name}")
    except Exception as e:
        logging.error(f"Model switch error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/info")
async def get_session_info(session_id: str):
    session = manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ingested_documents": session.rag.get_ingested_documents_info()}

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    manager.delete_session(session_id)
    return {"message": "Session deleted"}

@app.get("/models")
async def list_models():
    model_manager = ModelManager()
    return {"models": model_manager.list_models()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)