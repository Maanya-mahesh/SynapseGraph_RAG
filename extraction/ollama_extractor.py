import asyncio
import json
import logging
from typing import Optional
import aiohttp
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 1: Strict Pydantic V2 Data Schemas
class SPOTriple(BaseModel): #Schema -> single Subject-Predicate-Object triple."""
    subject: str = Field(..., description="Canonical source entity or concept in plain text.")
    predicate: str = Field(..., description="Directed relation in UPPER_SNAKE_CASE (e.g., PARTITIONS_INTO, USES_ALGORITHM).")
    object: str = Field(..., description="Canonical target entity, value, or concept.")
    confidence: float = Field(default=1.0, 
        ge=0.0, 
        le=1.0, 
        description="Confidence score assigned to extraction [0.0 - 1.0].")


class ExtractionResponse(BaseModel): # list of extracted SPO triples from a text chunk
    triples: list[SPOTriple] = Field( default_factory=list, 
        description="List of extracted SPOTriple objects.")


#2: Asynchronous Local Graph SPO Extractor
class LocalGraphSPOExtractor: #Asynchronous SPO extraction engine utilizing Ollama's /api/chat endpoint
    SYSTEM_INSTRUCTION = (
        "You are an expert Knowledge Graph Information Extraction Engine.\n"
        "Your task is to extract all core Subject-Predicate-Object (SPO) triples from the user's text.\n\n"
        "Strict Extraction Rules:\n"
        "1. Predicates MUST be concise, action-oriented UPPER_SNAKE_CASE verbs (e.g., USES_LIBRARY, DEPENDS_ON).\n"
        "2. Subjects and Objects must be explicit canonical entities or clear concepts.\n"
        "3. Do NOT invent unstated facts. Extract only what is directly supported by the text.\n"
        "4. Return strictly valid JSON adhering to the target schema."
    )

    def __init__(
        self,
        ollama_endpoint: str = "http://localhost:11434/api/chat",
        model_name: str = "llama3:8b",
        max_concurrency: int = 4,
        timeout_seconds: float = 30.0) -> None:
        self.endpoint = ollama_endpoint
        self.model_name = model_name
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self.session: Optional[aiohttp.ClientSession] = None

    async def initialize(self) -> None: # aiohttp ClientSession connection pool here
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info("Initialized aiohttp ClientSession connection pool.")

    async def extract_triples_from_chunk(self, text_chunk: str, max_retries: int = 3) -> list[SPOTriple]:
  # queries Ollama /api/chat asynchronously to extract SPO triples Uses system/user role separation
        if not text_chunk.strip():
            return []
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": self.SYSTEM_INSTRUCTION},
                {"role": "user", "content": f"Extract all relational SPO triples from this text:\n\"{text_chunk}\""}],
            "format": ExtractionResponse.model_json_schema(),  # Logit masking here
            "stream": False,
            "options": {"temperature": 0.0}  # temp/creativity 0 bcoz,-> deterministic
        }

        async with self.semaphore:
            assert self.session is not None, "ClientSession is uninitialized. Call initialize() first."

            for attempt in range(max_retries):
                try:
                    async with self.session.post(self.endpoint, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.warning(f"HTTP {response.status} from Ollama: {error_text}")
                            await asyncio.sleep(2 ** attempt)
                            continue

                        data = await response.json()
                        content = data.get("message", {}).get("content", "{}")
                        
                        parsed_json = json.loads(content)    # validating response against Pydantic
                        validated_response = ExtractionResponse.model_validate(parsed_json)
                        return validated_response.triples

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Attempt {attempt + 1} failed for chunk: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt) #exponential backoff here  {system repeatedly retries a failed operation by increasing the wait time exponentially after each attempt}
                except (json.JSONDecodeError, Exception) as e:
                    logger.error(f"Validation failure on response: {str(e)}")
                    return []

            logger.error(f"Failed to extract triples after {max_retries} attempts.")
            return []

    async def batch_extract(self, text_chunks: list[str]) -> list[list[SPOTriple]]:
        """Concurrently processes multiple document chunks using asyncio.gather."""
        tasks = [self.extract_triples_from_chunk(chunk) for chunk in text_chunks]
        return await asyncio.gather(*tasks)

    async def close(self) -> None:
        """Closes the underlying HTTP session pool."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Closed aiohttp ClientSession connection pool.")
