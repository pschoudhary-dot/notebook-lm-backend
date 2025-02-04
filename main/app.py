import asyncio
import logging
import os
from dotenv import load_dotenv
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
        logging.FileHandler('document_processor.log'),
        logging.StreamHandler()
    ]
)

load_dotenv()

async def process_and_ingest_files(processor, rag, file_paths):
    """
    Process the given files and ingest successful results.
    """
    try:
        # Process files with progress tracking
        results = await process_files_with_progress(processor, file_paths)
        
        # Filter out failed results and ingest successful ones
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        # Handle failed results
        if failed_results:
            print("\nFailed to process the following files:")
            for result in failed_results:
                print(f"- {result.metadata.get('path', 'unknown')}: {result.error_message}")
        
        # Handle successful results
        if successful_results:
            print("\nIngesting successfully processed documents...")
            await rag.ingest_documents(successful_results)
            print("\n" + rag.get_ingested_documents_info())
        else:
            print("\nNo documents were successfully processed for ingestion.")
            return successful_results
    except Exception as e:
        print(f"Error processing and ingesting files: {e}")
        logging.error("Error processing and ingesting files", exc_info=True)
        return []

async def handle_podcast_creation(rag, successful_results, podcast_processor):
    """
    Handle podcast creation from ingested content.
    """
    if not successful_results:
        print("No documents available for podcast creation!")
        return
    
    # Get podcast topic from user
    podcast_query = input("Enter podcast topic or query: ").strip()
    
    try:
        # Get relevant content from RAG
        podcast_content = await rag.query(podcast_query)
        
        # Generate transcript
        transcript = await podcast_processor.process_podcast(podcast_content)
        print("\nGenerated Transcript:\n")
        print(transcript)
        
        # Voice selection
        print("\nAvailable Voices:")
        for voice in VOICES:
            print(f"- {voice}")
        
        voice1 = input("Enter name for Host 1 voice: ").strip()
        voice2 = input("Enter name for Host 2 voice: ").strip()
        
        print("Generating podcast audio... hold tight!")

        # Generate audio
        audio_url = podcast_processor.generate_audio(
            transcript=transcript,
            host1_voice=voice1,
            host2_voice=voice2
        )
        print(f"\nPodcast audio generated successfully! URL: {audio_url}")
        
    except Exception as e:
        print(f"Podcast generation failed: {str(e)}")
        logging.error("Podcast generation failed", exc_info=True)

async def handle_model_switch(rag, successful_results, model_manager):
    """
    Allow switching models and re-ingesting documents.
    """
    print("\nAvailable Models:")
    for model in model_manager.list_models():
        print(f"- {model['name']} ({model['provider']}) - Cost: ${model['cost_per_million_tokens']} per million tokens")
    
    while True:
        new_model = input("\nEnter new model name: ").strip()
        if not new_model:
            print("Model name cannot be empty, try again.")
            continue

        try:
            # Switch to the new model
            rag = RAGSystem(new_model)
            # Re-ingest documents with new model
            await rag.ingest_documents(successful_results)
            print(f"\nSwitched to model: {new_model}")
            break
        except ValueError as e:
            print(f"Error: {e}")
            logging.error("Error switching model", exc_info=True)

async def main():
    try:
        processor = DocumentProcessor()
        model_manager = ModelManager()
        # Initialize PodcastProcessor with your credentials
        podcast_processor = PodcastProcessor(
            user_id="ssFAgjpDO5NVuWyC9AOLghsnTlS2",
            secret_key="ak-f16737b738ea4dd78ea3aefa81fb9c25"
        )
        
        print("Available Models:")
        for model in model_manager.list_models():
            print(f"- {model['name']} ({model['provider']}) - Cost: ${model['cost_per_million_tokens']} per million tokens")
        
        while True:
            model_name = input("\nEnter model name (or press Enter for default llama-3.3-70b-versatile): ").strip()
            if not model_name:
                model_name = "llama-3.3-70b-versatile"
            try:
                rag = RAGSystem(model_name)
                break
            except ValueError as e:
                print(f"Error: {e}")
                logging.error("Error initializing RAG system with model", exc_info=True)
        
        # Process files
        file_paths = [
            "https://genius.com/Lord-huron-mine-forever-lyrics",
            "./docs/video.mp4",
            "./docs/audio.mp3"
        ]

        # Allow user to add more files
        while True:
            additional_file = input("\nEnter additional file path (or press Enter to continue): ").strip()
            if not additional_file:
                break
            file_paths.append(additional_file)

        successful_results = await process_and_ingest_files(processor, rag, file_paths)

        # Chat interface
        print("\nRAG System Ready. Available commands:")
        print("- 'exit': Quit the program")
        print("- 'switch model': Change the language model")
        print("- 'info': Show ingested documents information")
        print("- 'create podcast': Generate a podcast from ingested content")
        print("- 'help': Show these commands")
        
        while True:
            query_input = input("\nQuestion: ").strip()
            
            if not query_input:
                continue
            
            if query_input.lower() == 'exit':
                break
            elif query_input.lower() == 'help':
                print("\nAvailable commands:")
                print("- 'exit': Quit the program")
                print("- 'switch model': Change the language model")
                print("- 'info': Show ingested documents information")
                print("- 'help': Show these commands")
                print("- 'create podcast': Generate a podcast from ingested content")

            elif query_input.lower() == 'create podcast':
                await handle_podcast_creation(rag, successful_results, podcast_processor)
            elif query_input.lower() == 'info':
                print("\n" + rag.get_ingested_documents_info())
            elif query_input.lower() == 'switch model':
                await handle_model_switch(rag, successful_results, model_manager)
            else:
                try:
                    response = await rag.query(query_input)
                    print(f"\nAnswer: {response}\n")
                except Exception as e:
                    print(f"\nError: Failed to process query - {str(e)}")
                    logging.error("Query processing error", exc_info=True)

    except Exception as e:
        print(f"\nCritical error: {str(e)}")
        logging.error("Critical error in main", exc_info=True)
    finally:
        # Cleanup temp directory
        try:
            for file in os.listdir(processor.tmp_dir):
                os.remove(os.path.join(processor.tmp_dir, file))
            os.rmdir(processor.tmp_dir)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary directory: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
        