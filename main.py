'''import os
import tempfile
import asyncio
import mimetypes
from docx import Document as DocxDocument
from dotenv import load_dotenv
from typing import List, Union
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import aiohttp
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
load_dotenv()


#1 create a class with the methods process_files, _process_pdf, _process_csv_excel, _process_doc, _process_image, _process_video, _process_audio, _process_url
class DocumentProcessor:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        self.browser_config = BrowserConfig(verbose=False)
        self.crawler_run_config = CrawlerRunConfig(
            word_count_threshold=50,
            remove_overlay_elements=True,
            process_iframes=True
        )

    async def process_file(self, file_path: str) -> str:
        if urlparse(file_path).scheme in ('http', 'https'):
            return await self._process_url(file_path)
        else:
            mime_type = self._get_mime_type(file_path)
            print(mime_type)
        if mime_type == "application/pdf":
            return self._process_pdf(file_path)
        elif mime_type in ["text/csv", "application/vnd.ms-excel"]:
            return self._process_csv_excel(file_path)
        elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            return self._process_doc(file_path)
        elif mime_type in ["doc/docx", "application/msword"]:
            return self._process_doc(file_path)
        elif mime_type in ["image/jpeg", "image/png", "image/jpg", "image/gif", "image/bmp", "image/tiff", "image/webp"]:
            return await self._process_image(file_path)
        elif mime_type in ["video/mp4", "video/avi", "video/mov", "video/wmv", "video/flv", "video/mkv"]:
            return await self._process_video(file_path)
        elif mime_type in ["audio/mpeg", "audio/wav"]:
            return await self._process_audio(file_path)
        elif mime_type == "text/plain":
            return await self._process_url(file_path)
        else:
            raise ValueError(f"Unsupported file type: {mime_type}")

    def _get_mime_type(self, file_path: str) -> str:
        return mimetypes.guess_type(file_path)[0]

    def _process_pdf(self, path: str) -> str:
        loader = PyPDFLoader(path)
        pages = loader.load()
        return "\n".join([p.page_content for p in pages])

    def _process_csv_excel(self, path: str) -> str:
        if path.endswith('.csv'):
            loader = CSVLoader(path)
        else:
            loader = UnstructuredExcelLoader(path)
        docs = loader.load()
        return "\n".join([d.page_content for d in docs])

    def _process_doc(self, path: str) -> str:
        try:
            doc = DocxDocument(path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            raise RuntimeError(f"Error processing document file: {str(e)}")

    async def _process_image(self, path: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        async with aiohttp.ClientSession() as session:
            with open(path, "rb") as f:
                data = f.read()
            async with session.post(API_URL, headers=self.headers, data=data) as response:
                result = await response.json()
                return result[0]['generated_text'] if isinstance(result, list) else result.get('error', 'Image processing failed')

    async def _process_video(self, path: str) -> str:
        try:
            # Create tmp directory if it doesn't exist
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            # Use a temporary file in our tmp directory
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False) as tmpfile:
                video = VideoFileClip(path)
                video.audio.write_audiofile(tmpfile.name)
                result = await self._process_audio(tmpfile.name)
                
                # Clean up the temporary file
                try:
                    os.unlink(tmpfile.name)
                except:
                    pass
                    
                return result
        except Exception as e:
            raise RuntimeError(f"Error processing video file: {str(e)}")

    async def _process_audio(self, path: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3-turbo"
        async with aiohttp.ClientSession() as session:
            with open(path, "rb") as f:
                data = f.read()
            async with session.post(API_URL, headers=self.headers, data=data) as response:
                result = await response.json()
                return result.get('text', 'Audio processing failed')

    async def _process_url(self, path: str) -> str:
        async with AsyncWebCrawler(config=self.browser_config) as crawler:
            result = await crawler.arun(
                url=path,
                config=self.crawler_run_config
            )
            return result.markdown if result.success else "Failed to crawl URL"

class RAGSystem:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        # self.llm = ChatGroq(temperature=0, model_name="llama3-70b-8192")
        self.llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

    async def ingest_documents(self, documents: List[Union[str, Document]]):
        if not documents:
            raise ValueError("No documents provided for ingestion")
        
        chunks = self.text_splitter.split_documents(
            [Document(page_content=doc) if isinstance(doc, str) else doc for doc in documents]
        )
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

    async def query(self, question: str, k: int = 5) -> str: #k is the number of kitne similar documents dekhna hai
        if self.vector_store is None:
            return "No documents loaded yet. Please ingest documents first."
        
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([d.page_content for d in docs])
        
        # prompt = ChatPromptTemplate.from_template(
        #     """Answer the question based only on the following context:
        #     {context}
        #     Question: {question}
        #     """
        # )
        prompt = ChatPromptTemplate.from_template(
            """
            ### Instructions
                    You are an intelligent RAG-based chatbot designed to assist users by providing detailed and relevant answers based on the provided context. 
                    You will answer questions by retrieving the most relevant documents and generating responses based solely on the retrieved information.

                    ### Context
                    {context}

                    ### User's Question
                    {question}

                    ### Response
                    Provide a comprehensive and well-structured response to the user's question based on the context. Ensure your answer is accurate, informative, and engaging.

                    ---
            """
        )
        chain = prompt | self.llm
        return (await chain.ainvoke({"context": context, "question": question})).content

async def main():
    processor = DocumentProcessor()
    rag = RAGSystem()

    documents = []
    file_paths = [
        "./docs/doc.pdf",
        "https://docs.crawl4ai.com/core/installation/",
        "https://docs.crawl4ai.com/core/fit-markdown/",
        "https://genius.com/Lord-huron-mine-forever-lyrics",
        "./docs/t.mp3",
        "./docs/LLM_Example.doc",
        "./docs/video.mp4",
        "./docs/audio.mp3"
    ]

    # Process files concurrently
    tasks = [processor.process_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            print(f"Error processing {path}: {str(result)}")
        else:
            documents.append(result)

    # Ingest documents
    await rag.ingest_documents(documents)

    # Chat interface
    print("RAG System Ready. Type 'exit' to quit.")
    while True:
        query_input = input("Question: ")
        if query_input.lower() == 'exit':
            break
        response = await rag.query(query_input)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())'''
import os
import tempfile
import asyncio
import mimetypes
from docx import Document as DocxDocument
from dotenv import load_dotenv
from typing import List, Union
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import aiohttp
from crawl4ai import AsyncWebCrawler
from crawl4ai.async_configs import BrowserConfig, CrawlerRunConfig
from moviepy.editor import VideoFileClip
from urllib.parse import urlparse
from models import ModelManager
load_dotenv()

#1 create a class with the methods process_files, _process_pdf, _process_csv_excel, _process_doc, _process_image, _process_video, _process_audio, _process_url
class DocumentProcessor:
    def __init__(self):
        self.headers = {"Authorization": f"Bearer {os.getenv('HF_API_KEY')}"}
        self.browser_config = BrowserConfig(verbose=False)
        self.crawler_run_config = CrawlerRunConfig(
            word_count_threshold=50,
            remove_overlay_elements=True,
            process_iframes=True
        )
        self.MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB for audio/video
        self.MAX_AUDIO_LENGTH = 120  # 2 minutes in seconds

    def _check_file_size(self, file_path: str) -> bool:
        if not os.path.isfile(file_path):
            return True  # Skip check for URLs
        
        mime_type = self._get_mime_type(file_path)
        if mime_type and (mime_type.startswith('audio/') or mime_type.startswith('video/')):
            size_limit = self.MAX_FILE_SIZE
        else:
            size_limit = self.MAX_FILE_SIZE * 2  # 20MB for other files
            
        return os.path.getsize(file_path) <= size_limit

    def _get_mime_type(self, file_path: str) -> str:
        return mimetypes.guess_type(file_path)[0]

    async def process_file(self, file_path: str) -> str:
        try:
            if urlparse(file_path).scheme in ('http', 'https'):
                result = await self._process_url(file_path)
                print(f"Successfully processed URL: {file_path}")
                return result
            
            if not self._check_file_size(file_path):
                print(f"Warning: File {file_path} exceeds size limit. Skipping processing.")
                return f"Error: File {os.path.basename(file_path)} exceeds size limit"

            mime_type = self._get_mime_type(file_path)
            print(f"Processing {file_path} ({mime_type})")

            result = None
            if mime_type == "application/pdf":
                result = self._process_pdf(file_path)
            elif mime_type in ["text/csv", "application/vnd.ms-excel"]:
                result = self._process_csv_excel(file_path)
            elif mime_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", "application/msword"]:
                result = self._process_doc(file_path)
            elif mime_type and mime_type.startswith('image/'):
                result = await self._process_image(file_path)
            elif mime_type and mime_type.startswith('video/'):
                result = await self._process_video(file_path)
            elif mime_type and mime_type.startswith('audio/'):
                result = await self._process_audio(file_path)
            elif mime_type == "text/plain":
                with open(file_path, 'r', encoding='utf-8') as f:
                    result = f.read()
            else:
                raise ValueError(f"Unsupported file type: {mime_type}")

            if result:
                print(f"Successfully processed: {file_path}")
                return result
            else:
                return f"Error: Could not process {os.path.basename(file_path)}"

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return f"Error processing {os.path.basename(file_path)}: {str(e)}"

    def _process_pdf(self, path: str) -> str:
        try:
            loader = PyPDFLoader(path)
            pages = loader.load()
            return "\n".join([p.page_content for p in pages])
        except Exception as e:
            raise RuntimeError(f"Error processing PDF file: {str(e)}")

    def _process_csv_excel(self, path: str) -> str:
        try:
            if path.endswith('.csv'):
                loader = CSVLoader(path)
            else:
                loader = UnstructuredExcelLoader(path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        except Exception as e:
            raise RuntimeError(f"Error processing CSV/Excel file: {str(e)}")

    def _process_doc(self, path: str) -> str:
        try:
            doc = DocxDocument(path)
            full_text = []
            for para in doc.paragraphs:
                full_text.append(para.text)
            return "\n".join(full_text)
        except Exception as e:
            raise RuntimeError(f"Error processing document file: {str(e)}")

    async def _process_image(self, path: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-large"
        try:
            async with aiohttp.ClientSession() as session:
                with open(path, "rb") as f:
                    data = f.read()
                async with session.post(API_URL, headers=self.headers, data=data) as response:
                    if response.status == 413:
                        return "Error: Image file is too large for processing"
                    result = await response.json()
                    return result[0]['generated_text'] if isinstance(result, list) else result.get('error', 'Image processing failed')
        except Exception as e:
            raise RuntimeError(f"Error processing image file: {str(e)}")

    async def _process_video(self, path: str) -> str:
        try:
            # Create tmp directory if it doesn't exist
            tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)
            
            video = VideoFileClip(path)
            if video.duration > self.MAX_AUDIO_LENGTH:
                video.close()
                return f"Error: Video duration exceeds {self.MAX_AUDIO_LENGTH} seconds limit"
            
            with tempfile.NamedTemporaryFile(suffix=".wav", dir=tmp_dir, delete=False) as tmpfile:
                video.audio.write_audiofile(tmpfile.name, logger=None)
                result = await self._process_audio(tmpfile.name)
                video.close()
                
                try:
                    os.unlink(tmpfile.name)
                except:
                    pass
                    
                return result
        except Exception as e:
            raise RuntimeError(f"Error processing video file: {str(e)}")

    async def _process_audio(self, path: str) -> str:
        API_URL = "https://api-inference.huggingface.co/models/openai/whisper-large-v3"
        try:
            # Check file size before processing
            if os.path.getsize(path) > self.MAX_FILE_SIZE:
                return "Error: Audio file is too large for processing"
                
            async with aiohttp.ClientSession() as session:
                with open(path, "rb") as f:
                    data = f.read()
                async with session.post(API_URL, headers=self.headers, data=data) as response:
                    if response.status == 413:
                        return "Error: Audio file is too large for API processing"
                    result = await response.json()
                    return result.get('text', 'Audio processing failed')
        except Exception as e:
            raise RuntimeError(f"Error processing audio file: {str(e)}")

    async def _process_url(self, path: str) -> str:
        try:
            async with AsyncWebCrawler(config=self.browser_config) as crawler:
                result = await crawler.arun(
                    url=path,
                    config=self.crawler_run_config
                )
                if result.success:
                    return result.markdown
                else:
                    raise RuntimeError("Failed to crawl URL")
        except Exception as e:
            raise RuntimeError(f"Error processing URL: {str(e)}") 

class RAGSystem:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.vector_store = None
        self.model_manager = ModelManager()
        self.llm = self.model_manager.get_model(model_name)

    async def ingest_documents(self, documents: List[Union[str, Document]]):
        if not documents:
            raise ValueError("No documents provided for ingestion")
        
        chunks = self.text_splitter.split_documents(
            [Document(page_content=doc) if isinstance(doc, str) else doc for doc in documents]
        )
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

    async def query(self, question: str, k: int = 5) -> str:
        if self.vector_store is None:
            return "No documents loaded yet. Please ingest documents first."
        
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([d.page_content for d in docs])
        
        prompt = ChatPromptTemplate.from_template(
            """
            ### Instructions
            You are an intelligent RAG-based chatbot designed to assist users by providing detailed and relevant answers based on the provided context. 
            You will answer questions by retrieving the most relevant documents and generating responses based solely on the retrieved information.

            ### Context
            {context}

            ### User's Question
            {question}

            ### Response
            Provide a comprehensive and well-structured response to the user's question based on the context. Ensure your answer is accurate, informative, and engaging.

            ---
            """
        )
        chain = prompt | self.llm
        return (await chain.ainvoke({"context": context, "question": question})).content

async def main():
    processor = DocumentProcessor()
    model_manager = ModelManager()
    
    # Display available models
    print("Available Models:")
    for model in model_manager.list_models():
        print(f"- {model['name']} ({model['provider']}) - Cost: ${model['cost_per_million_tokens']} per million tokens")
    
    # Let user select model
    while True:
        model_name = input("\nEnter model name (or press Enter for default llama-3.3-70b-versatile): ").strip()
        if not model_name:
            model_name = "llama-3.3-70b-versatile"
        try:
            rag = RAGSystem(model_name)
            break
        except ValueError as e:
            print(f"Error: {e}")
    
    documents = []
    file_paths = [
        "./docs/doc.pdf",
        "https://docs.crawl4ai.com/core/installation/",
        "https://docs.crawl4ai.com/core/fit-markdown/",
        "https://genius.com/Lord-huron-mine-forever-lyrics",
        "./docs/t.mp3",
        "./docs/LLM_Example.doc",
        "./docs/video.mp4",
        "./docs/audio.mp3"
    ]

    # Process files concurrently
    tasks = [processor.process_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    for path, result in zip(file_paths, results):
        if isinstance(result, Exception):
            print(f"Error processing {path}: {str(result)}")
        else:
            documents.append(result)

    # Ingest documents
    await rag.ingest_documents(documents)

    # Chat interface
    print("\nRAG System Ready. Type 'exit' to quit.")
    print("Type 'switch model' to change the model.")
    
    while True:
        query_input = input("\nQuestion: ")
        if query_input.lower() == 'exit':
            break
        elif query_input.lower() == 'switch model':
            print("\nAvailable Models:")
            for model in model_manager.list_models():
                print(f"- {model['name']} ({model['provider']}) - Cost: ${model['cost_per_million_tokens']} per million tokens")
            
            while True:
                new_model = input("\nEnter new model name: ").strip()
                try:
                    rag = RAGSystem(new_model)
                    print(f"\nSwitched to model: {new_model}")
                    break
                except ValueError as e:
                    print(f"Error: {e}")
            continue
        
        response = await rag.query(query_input)
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(main())