import os
import random
import json
from typing import List, Dict, Any, Optional
import openai
from openai import OpenAI
import requests
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class MultiAPILLMProcessor:
    """
    Multi-API LLM Processor that can use multiple LLM providers with fallback
    Compatible with your existing Decision Engine
    """
    
    def __init__(self):
        # Initialize multiple API clients
        self.apis = []
        self.current_api_index = 0
        
        # OpenAI API Keys (you can add multiple)
        openai_keys = [
            os.getenv('OPENAI_API_KEY'),      # Your main key
            os.getenv('OPENAI_API_KEY_1'),    # Backup keys
            os.getenv('OPENAI_API_KEY_2'),
            os.getenv('OPENAI_API_KEY_3'),
        ]
        
        # Add OpenAI clients
        for i, key in enumerate(openai_keys):
            if key:
                self.apis.append({
                    'type': 'openai',
                    'client': OpenAI(api_key=key),
                    'model': 'gpt-3.5-turbo',
                    'key': key,
                    'priority': 1  # Highest priority
                })
                logger.info(f"Added OpenAI API client #{i+1}")
        
        # Add Groq API (Free tier: 15 req/min, 6000 req/day)
        groq_key = os.getenv('GROQ_API_KEY')
        if groq_key:
            self.apis.append({
                'type': 'groq',
                'key': groq_key,
                'model': 'llama3-8b-8192',
                'base_url': 'https://api.groq.com/openai/v1',
                'priority': 2
            })
            logger.info("Added Groq API client")
        
        # Add Hugging Face API (Free tier available) - FIXED IMPLEMENTATION
        hf_key = os.getenv('HUGGINGFACE_API_KEY')
        if hf_key:
            self.apis.append({
                'type': 'huggingface',
                'key': hf_key,
                'model': 'microsoft/DialoGPT-large',  # Better model
                'priority': 3
            })
            logger.info("Added Hugging Face API client")
        
        # Add Cohere API (Free tier: 100 req/min)
        cohere_key = os.getenv('COHERE_API_KEY')
        if cohere_key:
            self.apis.append({
                'type': 'cohere',
                'key': cohere_key,
                'model': 'command-light',
                'priority': 4
            })
            logger.info("Added Cohere API client")
        
        # Sort APIs by priority
        self.apis.sort(key=lambda x: x['priority'])
        
        logger.info(f"Initialized with {len(self.apis)} API endpoints")
        if len(self.apis) == 0:
            logger.warning("No API keys found! Please add API keys to your .env file")
    
    # IMPORTANT: Keep the same method names as your original LLMProcessor
    # so your existing code doesn't break
    
    def parse_query(self, query: str) -> Dict[str, Any]:
        """
        Parse natural language query - SAME METHOD as your original
        """
        system_prompt = """You are an expert query parser for insurance, legal, HR, and compliance documents. 
        Parse the given query and extract structured information.
        
        Return a JSON object with:
        - intent: The main intent (e.g., "coverage_check", "eligibility", "claim_amount", "waiting_period")
        - entities: Extracted entities like procedures, amounts, time periods, conditions
        - keywords: Key terms for semantic search
        - domain: Document domain (insurance, legal, hr, compliance)
        - complexity: Query complexity (simple, medium, complex)
        
        Be precise and comprehensive in your extraction."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Parse this query: '{query}'"}
        ]
        
        try:
            response = self.generate_response(messages)
            
            # Try to parse JSON from response
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                # If JSON parsing fails, extract JSON from text
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError("No valid JSON found in response")
            
            logger.info(f"Successfully parsed query with intent: {result.get('intent', 'unknown')}")
            return result
        except Exception as e:
            logger.error(f"Error parsing query: {str(e)}")
            return self._fallback_parse_query(query)
    
    def generate_answer(self, question: str, context: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive answer - SAME METHOD as your original
        """
        # Prepare context from retrieved chunks
        context_text = self._prepare_context(retrieved_chunks)
        
        system_prompt = """You are an expert document analyzer specializing in insurance, legal, HR, and compliance documents.
        
        Given a question and relevant document excerpts, provide:
        1. A clear, accurate answer
        2. Specific clause references that support your answer
        3. Step-by-step reasoning
        4. Confidence level (high/medium/low)
        5. Any important conditions or limitations
        
        Be precise, cite specific clauses, and explain your reasoning clearly.
        If information is insufficient, state what additional information would be needed."""
        
        user_prompt = f"""Question: {question}
        
        Relevant Document Excerpts:
        {context_text}
        
        Provide a comprehensive answer with supporting evidence and reasoning."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            answer_text = self.generate_response(messages)
            
            return {
                "answer": answer_text,
                "reasoning": self._extract_reasoning(answer_text),
                "confidence": self._assess_confidence(retrieved_chunks),
                "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(retrieved_chunks[:3])],
                "token_usage": len(answer_text) // 4  # Rough estimate
            }
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return self._fallback_generate_answer(question, retrieved_chunks)
    
    def generate_response(self, messages: List[Dict], max_retries: int = 2) -> str:
        """
        Generate response using available APIs with fallback mechanism
        """
        if not self.apis:
            raise Exception("No API endpoints configured. Please add API keys to your .env file")
        
        # Try APIs in priority order (don't shuffle to maintain reliability)
        last_error = None
        
        for api_config in self.apis:
            for attempt in range(max_retries):
                try:
                    logger.debug(f"Trying {api_config['type']} API (attempt {attempt + 1})")
                    
                    if api_config['type'] == 'openai':
                        return self._call_openai(api_config, messages)
                    elif api_config['type'] == 'groq':
                        return self._call_groq(api_config, messages)
                    elif api_config['type'] == 'huggingface':
                        return self._call_huggingface(api_config, messages)
                    elif api_config['type'] == 'cohere':
                        return self._call_cohere(api_config, messages)
                    
                except Exception as e:
                    last_error = e
                    logger.warning(f"Error with {api_config['type']}: {str(e)}")
                    
                    # Wait before retry (shorter wait for faster fallback)
                    if attempt < max_retries - 1:
                        time.sleep(1 + attempt)  # 1s, 2s wait
                    continue
            
            logger.info(f"Failed all attempts for {api_config['type']}, trying next API")
        
        raise Exception(f"All API endpoints failed. Last error: {str(last_error)}")
    
    def _call_openai(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call OpenAI API"""
        response = api_config['client'].chat.completions.create(
            model=api_config['model'],
            messages=messages,
            max_tokens=1000,
            temperature=0.2,
            timeout=30
        )
        return response.choices[0].message.content
    
    def _call_groq(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Groq API (OpenAI-compatible)"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': api_config['model'],
            'messages': messages,
            'max_tokens': 1000,
            'temperature': 0.2
        }
        
        response = requests.post(
            f"{api_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    
    def _call_huggingface(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Hugging Face API - IMPROVED IMPLEMENTATION"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        # Convert messages to a single prompt for HF
        prompt = self._messages_to_prompt(messages)
        
        # Try conversational model first
        try:
            data = {
                'inputs': {
                    'past_user_inputs': [],
                    'generated_responses': [],
                    'text': prompt
                },
                'parameters': {
                    'max_length': 1000,
                    'temperature': 0.2,
                    'do_sample': True,
                    'top_p': 0.9
                },
                'options': {
                    'wait_for_model': True,
                    'use_cache': False
                }
            }
            
            response = requests.post(
                f"https://api-inference.huggingface.co/models/{api_config['model']}",
                headers=headers,
                json=data,
                timeout=60  # HF can be slow
            )
            
            if response.status_code == 200:
                result = response.json()
                if 'generated_text' in result:
                    return result['generated_text']
                elif 'conversation' in result:
                    return result['conversation']['generated_responses'][-1]
                else:
                    return str(result)
            else:
                raise Exception(f"HuggingFace API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            # Fallback to text generation model
            logger.warning(f"Conversational model failed, trying text generation: {e}")
            return self._call_huggingface_text_generation(api_config, prompt, headers)
    
    def _call_huggingface_text_generation(self, api_config: Dict, prompt: str, headers: Dict) -> str:
        """Fallback to text generation model"""
        # Use a more reliable text generation model
        fallback_model = "microsoft/DialoGPT-medium"
        
        data = {
            'inputs': prompt,
            'parameters': {
                'max_new_tokens': 500,
                'temperature': 0.2,
                'do_sample': True,
                'return_full_text': False
            },
            'options': {
                'wait_for_model': True
            }
        }
        
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{fallback_model}",
            headers=headers,
            json=data,
            timeout=60
        )
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get('generated_text', '')
                # Clean up the response
                return generated.replace(prompt, '').strip()
            return str(result)
        else:
            raise Exception(f"HuggingFace fallback error: {response.status_code} - {response.text}")
    
    def _call_cohere(self, api_config: Dict, messages: List[Dict]) -> str:
        """Call Cohere API"""
        headers = {
            'Authorization': f'Bearer {api_config["key"]}',
            'Content-Type': 'application/json'
        }
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        data = {
            'model': api_config['model'],
            'prompt': prompt,
            'max_tokens': 1000,
            'temperature': 0.2,
            'stop_sequences': []
        }
        
        response = requests.post(
            'https://api.cohere.ai/v1/generate',
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()['generations'][0]['text'].strip()
    
    def _messages_to_prompt(self, messages: List[Dict]) -> str:
        """Convert OpenAI-style messages to a single prompt"""
        prompt_parts = []
        
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)
    
    # Helper methods (same as your original)
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from retrieved chunks for LLM input"""
        if not chunks:
            return "No relevant document excerpts found."
        
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):
            relevance = chunk.get('relevance_score', 0)
            context_parts.append(f"[Excerpt {i+1}] (Relevance: {relevance:.3f})")
            context_parts.append(chunk.get('text', ''))
            context_parts.append("")
        return "\n".join(context_parts)
    
    def _extract_reasoning(self, answer_text: str) -> str:
        """Extract reasoning section from LLM response"""
        lines = answer_text.split('\n')
        reasoning_lines = []
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['because', 'since', 'according to', 'based on', 'reasoning']):
                reasoning_lines.append(line)
        
        return ' '.join(reasoning_lines) if reasoning_lines else "Reasoning not explicitly provided."
    
    def _assess_confidence(self, chunks: List[Dict]) -> str:
        """Assess confidence based on retrieved chunks quality"""
        if not chunks:
            return "low"
        
        avg_score = sum(chunk.get('relevance_score', 0) for chunk in chunks) / len(chunks)
        
        if avg_score > 0.8:
            return "high"
        elif avg_score > 0.6:
            return "medium"
        else:
            return "low"
    
    def _fallback_parse_query(self, query: str) -> Dict[str, Any]:
        """Fallback query parsing without LLM"""
        import re
        
        entities = {}
        
        # Extract common insurance/medical entities
        if re.search(r'surgery|operation|procedure', query.lower()):
            entities['procedure_type'] = 'surgery'
        
        if re.search(r'waiting period|wait', query.lower()):
            entities['concern'] = 'waiting_period'
        
        if re.search(r'cover|coverage', query.lower()):
            entities['concern'] = 'coverage'
        
        if re.search(r'claim|reimbursemen', query.lower()):
            entities['concern'] = 'claim'
        
        # Extract numbers (amounts, ages, periods)
        numbers = re.findall(r'\d+', query)
        if numbers:
            entities['numbers'] = numbers
        
        # Extract age
        age_match = re.search(r'(\d+)\s*year', query.lower())
        if age_match:
            entities['age'] = int(age_match.group(1))
        
        return {
            "intent": "coverage_check",
            "entities": entities,
            "keywords": query.lower().split(),
            "domain": "insurance",
            "complexity": "medium"
        }
    
    def _fallback_generate_answer(self, question: str, chunks: List[Dict]) -> Dict[str, Any]:
        """Fallback answer generation without LLM"""
        if not chunks:
            return {
                "answer": "Unable to find relevant information in the document to answer this question.",
                "reasoning": "No relevant chunks retrieved from the document.",
                "confidence": "low",
                "supporting_chunks": [],
                "token_usage": 0
            }
        
        best_chunk = chunks[0]
        chunk_text = best_chunk.get('text', '')
        
        return {
            "answer": f"Based on the document excerpt: {chunk_text[:300]}{'...' if len(chunk_text) > 300 else ''}",
            "reasoning": f"Answer derived from highest relevance chunk (score: {best_chunk.get('relevance_score', 0):.3f})",
            "confidence": self._assess_confidence(chunks),
            "supporting_chunks": [chunk.get('id', i) for i, chunk in enumerate(chunks[:3])],
            "token_usage": 0
        }

# For backward compatibility with your existing code
LLMProcessor = MultiAPILLMProcessor