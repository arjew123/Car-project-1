import os
import asyncio
from fastapi import APIRouter, HTTPException
from google import genai
from pydantic import BaseModel, Field

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class ChatRequest(BaseModel):
    message: str

class CarChatFeatures(BaseModel):
    brand: str = Field(description="Vehicle manufacturer name")
    model: str = Field(description="Specific vehicle model")
    year: int = Field(description="Model year")
    mileage: int = Field(description="Odometer reading in kilometers or miles")
    condition: str = Field(description="Condition rating: Excellent, Good, Fair, or Poor")

@router.post("/api/chat-valuation")
async def chat_valuation(payload: ChatRequest):
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    system_prompt = (
        "You are an automotive appraisal assistant. Parse the user's natural language vehicle description "
        "and extract: brand, model, year, mileage, and condition. If mileage is not mentioned, infer a reasonable "
        "estimate assuming 12,000 to 15,000 km/miles per year. If condition is not mentioned, default to 'Good'."
    )

    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            # Switched to aio (async) client
            response = await client.aio.models.generate_content(
                model="gemini-3.8-flash",
                contents=[system_prompt, payload.message],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CarChatFeatures,
                    temperature=0.2
                )
            )
            return response.parsed.model_dump()
            
        except Exception as e:
            error_msg = str(e)
            
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise HTTPException(status_code=429, detail="Daily AI request limit reached. Please generate a new API key.")

            if "503" in error_msg and attempt < max_retries - 1:
                wait_time = 4 * (attempt + 1) 
                print(f"Chat API busy, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
                
            print(f"Chat API Error on attempt {attempt + 1}: {error_msg}")
            raise HTTPException(status_code=500, detail="Google AI servers are currently overloaded. Please try again in a few minutes.")