import logging
import os
from typing import List
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
from ProcessingResult import ProcessingResult
from models import ModelManager
import logging
from DocumentProcessor import DocumentProcessor


class RAGSystem:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        self.vector_store = None
        self.model_manager = ModelManager()
        self.llm = self.model_manager.get_model(model_name)
        self.document_metadata = []

    async def ingest_documents(self, documents: List[ProcessingResult]):
        if not documents:
            raise ValueError("No documents provided for ingestion")
        
        processed_docs = []
        for doc in documents:
            if doc.success and doc.content:
                doc_chunks = self.text_splitter.split_text(doc.content)
                processed_docs.extend([
                    Document(
                        page_content=chunk,
                        metadata=doc.metadata
                    ) for chunk in doc_chunks
                ])
                self.document_metadata.append(doc.metadata)
        
        if not processed_docs:
            raise ValueError("No valid documents to process")
        
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(processed_docs, self.embeddings)
        else:
            self.vector_store.add_documents(processed_docs)

    def get_ingested_documents_info(self) -> str:
        if not self.document_metadata:
            return "No documents have been ingested yet."
        
        info = ["Ingested documents:"]
        for meta in self.document_metadata:
            doc_type = meta.get('type', 'unknown')
            path = meta.get('path', 'unknown')
            info.append(f"- {os.path.basename(path)} (Type: {doc_type})")
        return "\n".join(info)

    async def query(self, question: str, k: int = 5) -> str:
        if self.vector_store is None:
            return "No documents loaded yet. Please ingest documents first."
        
        docs = self.vector_store.similarity_search(question, k=k)
        context = "\n\n".join([
            f"[Source: {os.path.basename(d.metadata.get('path', 'unknown'))}]\n{d.page_content}"
            for d in docs
        ])
        
        prompt = ChatPromptTemplate.from_template(
            """
            ### Instructions
            You are an intelligent RAG-based chatbot designed to assist users by providing detailed and relevant answers based on the provided context. 
            You will answer questions by retrieving the most relevant documents and generating responses based solely on the retrieved information.
            When referencing information, mention the source document in your response.

            ### Context
            {context}

            ### User's Question
            {question}

            ### Response
            Please provide a comprehensive and well-structured response to the user's question based on the context. 
            Ensure your answer is accurate, informative, and properly cites the sources used.
            If the question cannot be fully answered using the provided context, acknowledge this limitation.

            ---
            """
        )
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({"context": context, "question": question})
            return response.content
        except Exception as e:
            logging.error(f"Error during query processing: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"

async def process_files_with_progress(processor: DocumentProcessor, file_paths: List[str]) -> List[ProcessingResult]:
    total_files = len(file_paths)
    processed = 0
    results = []
    
    print(f"\nProcessing {total_files} files...")
    
    for path in file_paths:
        processed += 1
        print(f"Processing ({processed}/{total_files}): {os.path.basename(path)}")
        result = await processor.process_file(path)
        results.append(result)
        
        if result.success:
            print(f"✓ Success: {os.path.basename(path)}")
        else:
            print(f"✗ Failed: {os.path.basename(path)} - {result.error_message}")
    
    return results