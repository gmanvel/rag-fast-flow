"""
LLM Service for interacting with Ollama-hosted Mistral model.
Provides both pure and RAG-enhanced query capabilities.
"""

import os
import requests
from typing import Optional


class LLMService:
    """Service for interacting with Ollama LLM API."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: str = "mistral"
    ):
        """
        Initialize LLM service.

        Args:
            base_url: Ollama API base URL (default: http://host.docker.internal:11434)
            model: Model name to use (default: mistral)
        """
        self.base_url = base_url or os.getenv(
            "OLLAMA_BASE_URL",
            "http://host.docker.internal:11434"
        )
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

    def query_pure(self, question: str) -> str:
        """
        Send a pure question to the LLM without any additional context.

        Args:
            question: User's question

        Returns:
            LLM's response
        """
        prompt = f"""You are an expert in Fast Flow methodologies including Wardley Mapping, Domain-Driven Design (DDD), and Team Topologies.

User Question: {question}

Please provide a helpful and accurate answer based on your knowledge."""

        return self._generate(prompt)

    def query_rag(self, question: str, context: str) -> str:
        """
        Send a question to the LLM with retrieved context (RAG approach).

        Args:
            question: User's question
            context: Retrieved context from vector database

        Returns:
            LLM's response enhanced with context
        """
        prompt = f"""You are an expert in Fast Flow methodologies including Wardley Mapping, Domain-Driven Design (DDD), and Team Topologies.

Use the following context from the Fast Flow documentation to answer the user's question. If the context is relevant, incorporate it into your answer. If the context doesn't fully answer the question, you can supplement with your general knowledge but prioritize the provided context.

Context:
{context}

User Question: {question}

Please provide a helpful and accurate answer."""

        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        """
        Internal method to generate response from Ollama.

        Args:
            prompt: Complete prompt to send to LLM

        Returns:
            Generated response
        """
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            result = response.json()
            return result.get("response", "No response generated.")

        except requests.exceptions.RequestException as e:
            return f"Error communicating with LLM: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"
