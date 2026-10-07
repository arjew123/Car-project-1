import os
import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from google import genai
from pydantic import BaseModel, Field

router = APIRouter()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class CarVisionFeatures(BaseModel):
    brand: str = Field(description="Vehicle manufacturer name, e.g., Toyota, Honda, Ford")
    model: str = Field(description="Vehicle model name, e.g., Camry, Civic, F-150")
    year: int = Field(description="Estimated manufacturing year based on vehicle generation and facelift")
    mileage: int = Field(description="Estimated average odometer reading if unknown (assume 12000 per year of age)")
    condition: str = Field(description="Exterior condition rating: Excellent, Good, Fair, or Poor")

@router.post("/api/extract-car-details")
async def extract_car_details(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    image_bytes = await file.read()
    prompt = (
        "Identify the car in this photo. Determine the brand/make, model, estimated model year "
        "(based on body style, headlamps, and generation), an estimated mileage, and condition."
    )

    max_retries = 3 # Reduced to fail faster if the server is truly gridlocked
    
    for attempt in range(max_retries):
        try:
            # Switched to aio (async) client to prevent thread blocking
            response = await client.aio.models.generate_content(
                model="gemini-3.8-flash",
                contents=[
                    prompt,
                    genai.types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
                ],
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=CarVisionFeatures,
                    temperature=0.1
                )
            )
            return response.parsed.model_dump()
            
        except Exception as e:
            error_msg = str(e)
            
            # Catch daily quota limits instantly
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                raise HTTPException(status_code=429, detail="Daily AI request limit reached. Please generate a new API key.")

            if "503" in error_msg and attempt < max_retries - 1:
                wait_time = 4 * (attempt + 1) 
                print(f"Vision API busy, retrying in {wait_time} seconds...")
                await asyncio.sleep(wait_time)
                continue
                
            print(f"Vision API Error on attempt {attempt + 1}: {error_msg}")
            raise HTTPException(status_code=500, detail="Google AI servers are currently overloaded. Please try again in a few minutes.")