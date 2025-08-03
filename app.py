import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import requests
import tempfile
from typing import List, Dict, Any
from extract import extract_text_from_pdf
from search import DocumentProcessor, SemanticSearch
from llm_processor import LLMProcessor  # Updated import
from decision_engine import DecisionEngine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM-Powered Query-Retrieval System", version="1.0.0")

# Global instances - Initialize with error handling
try:
    document_processor = DocumentProcessor()
    semantic_search = SemanticSearch()
    llm_processor = LLMProcessor()
    decision_engine = DecisionEngine(llm_processor)
    logger.info("Successfully initialized all components")
except Exception as e:
    logger.error(f"Failed to initialize components: {str(e)}")
    raise

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

class QueryResponse(BaseModel):
    answers: List[str]

@app.post("/hackrx/run", response_model=QueryResponse)
async def run_query(req: QueryRequest, authorization: str = Header(None)):
    """
    Main endpoint for processing document queries using LLM-powered retrieval system
    """
    # Validate authorization
    expected_token = f"Bearer {os.getenv('BEARER_TOKEN', '16ca23504efb8f8b98b1d84b2516a4b6ccb69f3c955ac9a8107497f5d14d6dbb')}"
    if not authorization or authorization != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authorization token")
    
    try:
        logger.info(f"Processing {len(req.questions)} questions for document: {req.documents}")
        
        # Step 1: Download and extract document content
        document_content = await download_and_extract_document(req.documents)
        
        # Step 2: Process document into chunks and build search index
        chunks = document_processor.create_chunks(document_content)
        search_index = semantic_search.build_index(chunks)
        
        # Step 3: Process each question
        answers = []
        for i, question in enumerate(req.questions):
            try:
                logger.info(f"Processing question {i+1}: {question[:50]}...")
                
                # Parse and understand the query using LLM
                parsed_query = llm_processor.parse_query(question)
                
                # Retrieve relevant chunks using semantic search
                relevant_chunks = semantic_search.search(search_index, question, chunks, top_k=5)
                
                # Generate answer with explainable reasoning
                answer_result = decision_engine.generate_answer(
                    question=question,
                    parsed_query=parsed_query,
                    context_chunks=relevant_chunks
                )
                
                # Extract just the answer text for the response
                answer_text = answer_result.get("answer", "Unable to generate answer")
                answers.append(answer_text)
                
                logger.info(f"Successfully processed question {i+1}")
                
            except Exception as e:
                logger.error(f"Error processing question {i+1}: {str(e)}")
                error_msg = f"Unable to process question due to: {str(e)[:100]}"
                answers.append(error_msg)
        
        logger.info(f"Successfully processed all {len(req.questions)} questions")
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in run_query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

async def download_and_extract_document(document_url: str) -> str:
    """
    Download document from URL and extract text content
    """
    try:
        logger.info(f"Downloading document from: {document_url}")
        
        # Download document with better error handling
        response = requests.get(document_url, timeout=60, stream=True)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            # Write in chunks to handle large files
            for chunk in response.iter_content(chunk_size=8192):
                temp_file.write(chunk)
            temp_path = temp_file.name
        
        try:
            # Extract text
            document_content = extract_text_from_pdf(temp_path)
            if not document_content.strip():
                raise ValueError("No text content extracted from document")
            
            logger.info(f"Extracted {len(document_content)} characters from document")
            return document_content
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except requests.RequestException as e:
        logger.error(f"Failed to download document: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to download document: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to process document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process document: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint with component status"""
    try:
        # Test if LLM processor is working
        test_result = llm_processor._fallback_parse_query("test query")
        
        return {
            "status": "healthy", 
            "service": "LLM Query-Retrieval System",
            "components": {
                "document_processor": "healthy",
                "semantic_search": "healthy", 
                "llm_processor": "healthy" if test_result else "degraded",
                "decision_engine": "healthy"
            },
            "api_endpoints": len(llm_processor.apis) if hasattr(llm_processor, 'apis') else 0
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "degraded",
            "service": "LLM Query-Retrieval System", 
            "error": str(e)
        }

@app.get("/api-status")
async def api_status():
    """Check status of available LLM APIs"""
    if not hasattr(llm_processor, 'apis'):
        return {"error": "LLM processor not properly initialized"}
    
    api_status = []
    for api in llm_processor.apis:
        api_status.append({
            "type": api['type'],
            "model": api['model'],
            "priority": api['priority'],
            "configured": True
        })
    
    return {
        "total_apis": len(llm_processor.apis),
        "apis": api_status,
        "fallback_available": len(llm_processor.apis) > 1
    }

if __name__ == "__main__":
    import uvicorn
    
    # Check if we have any API keys configured
    if not hasattr(llm_processor, 'apis') or len(llm_processor.apis) == 0:
        logger.warning("⚠️  No LLM API keys found! Please configure your .env file.")
        logger.warning("The server will start but requests will fail until API keys are added.")
    else:
        logger.info(f"✅ Server starting with {len(llm_processor.apis)} LLM API(s) configured")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)