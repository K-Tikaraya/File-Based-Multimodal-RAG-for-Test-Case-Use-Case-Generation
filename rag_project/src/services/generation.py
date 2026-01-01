import os
import json
import re
import sys
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.logger import setup_logger
from core.config import GROQ_API_KEY

logger = setup_logger()

# --- Data Models ---
class TestCase(BaseModel):
    title: str = Field(..., description="Title of the test case")
    goal: str = Field(..., description="The primary goal of this test case")
    preconditions: str = Field(..., description="Preconditions required before execution")
    test_data: Optional[str] = Field(None, description="Specific test data used (e.g., email='foo@bar.com')")
    steps: List[str] = Field(..., description="Step-by-step instructions")
    expected_results: str = Field(..., description="Expected outcomes")
    type: Literal["positive", "negative", "boundary"] = Field(..., description="Type of test case")
    negative_cases: Optional[List[str]] = Field(default=[], description="Specific negative scenarios if applicable")
    boundary_cases: Optional[List[str]] = Field(default=[], description="Edge cases or boundary conditions")

class TestSuite(BaseModel):
    test_cases: List[TestCase] = Field(..., description="List of generated test cases")
    status: Optional[str] = Field(None, description="Status code, e.g., 'missing_info'")
    missing_info_questions: Optional[List[str]] = Field(None, description="Specific questions if context is missing")

# --- Generator Class ---
class GenerationEngine:
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", guardrail_model: Optional[str] = None):
        self.api_key = GROQ_API_KEY
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is missing. Please set it in your .env file.")
            
        self.output_model = model_name
        self.guardrail_model = guardrail_model
        
        # Main LLM for generation
        self.llm = ChatGroq(
            model=model_name, 
            api_key=self.api_key,
            temperature=0.1
        )
        
        # Guardrail LLM (if selected)
        if self.guardrail_model and self.guardrail_model.lower() != "none":
            self.guard_llm = ChatGroq(
                model=self.guardrail_model,
                api_key=self.api_key,
                temperature=0.0
            )
        else:
            self.guard_llm = None
        
        self.parser = PydanticOutputParser(pydantic_object=TestSuite)

    def _check_safety(self, query: str) -> bool:
        """
        Checks if the query is safe using the guardrail model.
        Returns True if safe, False if unsafe.
        """
        if not self.guard_llm:
            return True
            
        try:
            # Simple prompt for checking safety. 
            # Note: Specialized Llama Guard usage typically uses formatted prompts.
            # Here we rely on the model's instruction following to classify.
            messages = [
                ("system", "You are a safety classification system. Analyze the user prompt. If it is safe and related to software development/testing, reply 'SAFE'. If it is harmful, illegal, or malicious, reply 'UNSAFE'."),
                ("user", query)
            ]
            response = self.guard_llm.invoke(messages)
            content = response.content.strip().upper()
            
            logger.info(f"Guardrail Check ({self.guardrail_model}): {content}")
            
            if "UNSAFE" in content:
                return False
            return True
            
        except Exception as e:
            logger.error(f"Guardrail check failed: {e}")
            # Fail open or closed? Let's fail open but warn.
            logger.warning("Proceeding despite guardrail error.")
            return True

    def generate(self, query: str, context_chunks: List[str]) -> TestSuite:
        # 1. Guardrail Check
        if not self._check_safety(query):
            return TestSuite(
                test_cases=[], 
                status="unsafe", 
                missing_info_questions=["The request was blocked by the safety guardrail."]
            )

        context_text = "\n\n".join(context_chunks)
        
        if not context_chunks:
            return TestSuite(test_cases=[], status="missing_info", missing_info_questions=["No context provided."])

        system_prompt = """You are a QA Test Engineer. Use ONLY the provided context. Do not hallucinate features.
        
        Required Output Structure:
        - Use Case Title
        - Goal
        - Preconditions
        - Test Data (if applicable)
        - Steps
        - Expected Results
        - Negative cases
        - Boundary cases
        
        If context is missing, return a JSON with a field `status: missing_info` and specific questions in `missing_info_questions`.
        Output MUST follow the JSON schema provided.
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context:\n{context}\n\nQuery: {query}\n\n{format_instructions}")
        ])
        
        chain = prompt | self.llm
        
        try:
            logger.info(f"Sending query to Groq LLM ({self.output_model})...")
            response = chain.invoke({
                "context": context_text, 
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
            
            content = response.content
            
            # Cleanup <think> tags
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            
            # Parse
            try:
                return self.parser.parse(content)
            except Exception:
                # Fallback JSON regex
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    data = json.loads(json_str)
                    return TestSuite(**data)
                else:
                    raise ValueError("No JSON found in response")
            
        except Exception as e:
            logger.error(f"Generation Error: {e}")
            return TestSuite(
                test_cases=[], 
                status="error", 
                missing_info_questions=[f"Error during generation: {str(e)}"]
            )
