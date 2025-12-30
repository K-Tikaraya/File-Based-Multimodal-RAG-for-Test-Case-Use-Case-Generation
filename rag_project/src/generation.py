import os
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv

load_dotenv()

# --- Data Models ---
class TestCase(BaseModel):
    title: str = Field(..., description="Title of the use case / test case")
    goal: str = Field(..., description="The primary goal of this test case")
    preconditions: List[str] = Field(..., description="List of preconditions required before execution")
    steps: List[str] = Field(..., description="Step-by-step instructions")
    expected_results: List[str] = Field(..., description="Expected outcomes")
    negative_cases: Optional[List[str]] = Field(default=[], description="Negative or boundary test scenarios related to this case")

class TestSuite(BaseModel):
    use_cases: List[TestCase] = Field(..., description="List of generated use cases")
    missing_information: Optional[str] = Field(None, description="Explicitly state any missing information or assumptions made if context was insufficient")

# --- Generator Class ---
class GenerationEngine:
    def __init__(self, model_name: str = "gpt-4o-mini", provider: str = "openai", api_key: str = None):
        self.provider = provider.lower()
        
        if self.provider == "local" or "ollama" in self.provider:
            # Assumes user has 'llama3' or similar pulled in Ollama
            self.llm = ChatOllama(model="llama3", format="json")
        elif self.provider == "groq":
            if not api_key:
                 api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("Groq API Key is required.")
            
            # Default to a good Groq model if generic name is passed, or use passed model
            # typical groq models: llama3-70b-8192, mixtral-8x7b-32768
            if model_name == "gpt-4o-mini": 
                model_name = "llama-3.1-8b-instant" 
                
            self.llm = ChatGroq(model=model_name, api_key=api_key)
        else:
            # OpenAI
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                print("Warning: OPENAI_API_KEY not found. Switching to Local (Ollama).")
                self.provider = "local"
                self.llm = ChatOllama(model="llama3", format="json")
            else:
                self.llm = ChatOpenAI(model=model_name, temperature=0.2, api_key=api_key)
        
        self.parser = PydanticOutputParser(pydantic_object=TestSuite)

    def generate(self, query: str, context_chunks: List[str]) -> TestSuite:
        context_text = "\n\n".join(context_chunks)
        
        system_prompt = """You are an expert QA Engineer and Product Manager. 
        Your task is to generate detailed Use Cases and Test Cases based strictly on the provided Context.
        
        Guidelines:
        1. GROUNDING: Only use information from the Context. Do not invent features not mentioned.
        2. STRUCTURE: Output must be a valid JSON object matching the requested schema.
        3. NEGATIVE CASES: Always think about edge cases, boundaries, and negative flows.
        4. MISSING INFO: If the context is insufficient to answer the query, populate the 'missing_information' field and provide as much as you can.
        
        Output Schema:
        {format_instructions}
        """
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "Context:\n{context}\n\nQuery: {query}")
        ])
        
        chain = prompt | self.llm | self.parser
        
        try:
            return chain.invoke({
                "context": context_text, 
                "query": query,
                "format_instructions": self.parser.get_format_instructions()
            })
        except Exception as e:
            print(f"Generation Error: {e}")
            # Return a safe fallback or re-raise
            return TestSuite(use_cases=[], missing_information=f"Error during generation: {str(e)}")

