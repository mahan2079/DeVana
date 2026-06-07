import os
import json
from typing import Dict, Any

from PyQt5.QtCore import QThread, pyqtSignal

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

class AIWorker(QThread):
    """
    Worker thread for non-blocking LLM API calls.
    Specifically handles Gemini API interactions with RAG context.
    """
    response_received = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    thinking_started = pyqtSignal()

    def __init__(self, api_key: str, model_name: str, system_prompt: str, user_query: str, context_data: Dict[str, Any]):
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_query = user_query
        self.context_data = context_data

    def run(self):
        if not GENAI_AVAILABLE:
            self.error_occurred.emit("New Gemini library (google-genai) not installed. Please run: pip install google-genai")
            return

        if not self.api_key:
            self.error_occurred.emit("API Key is missing. Please add it in the Assistant Settings.")
            return

        self.thinking_started.emit()

        try:
            # 1. Configure GenAI Client
            client = genai.Client(api_key=self.api_key)

            # 2. Enrich query with context data (The "State Bridge")
            context_str = json.dumps(self.context_data, indent=2)
            full_prompt = f"### CURRENT SYSTEM STATE ###\n{context_str}\n\n### USER QUERY ###\n{self.user_query}"

            # 3. Call API using new SDK structure
            response = client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "system_instruction": self.system_prompt
                }
            )
            
            if response and response.text:
                self.response_received.emit(response.text)
            else:
                self.error_occurred.emit("Empty response received from Gemini.")

        except Exception as e:
            self.error_occurred.emit(f"AI Worker Error: {str(e)}")

class RAGHelper:
    """
    Utility to find and load relevant documentation snippets for the LLM context.
    """
    @staticmethod
    def get_relevant_docs(query: str, doc_dir: str = "Documents") -> str:
        # For a basic implementation, we can scan the INDEX.md or specific keywords.
        # Future: Use vector embeddings here.
        context_snippets = []
        
        # Keywords for routing
        keywords = {
            "ga": "Algorithms/GA_Deep_Spec.md",
            "pso": "Algorithms/PSO.md",
            "pinn": "Analysis/PINN_Deep_Spec.md",
            "frf": "Analysis/FRF_Deep_Spec.md",
            "sobol": "Analysis/Sobol.md",
            "cost": "Algorithms/ObjectiveFunction.md"
        }

        query_lower = query.lower()
        for key, path in keywords.items():
            if key in query_lower:
                full_path = os.path.join(doc_dir, path)
                if os.path.exists(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                            # Strip frontmatter for token efficiency
                            if content.startswith("---"):
                                parts = content.split("---", 2)
                                if len(parts) > 2:
                                    content = parts[2]
                            context_snippets.append(f"--- DOCUMENT: {path} ---\n{content}")
                    except Exception:
                        pass

        return "\n\n".join(context_snippets) if context_snippets else "No specific documentation found for this query."
