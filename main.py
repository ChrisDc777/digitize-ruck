from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import google.generativeai as genai
import pandas as pd
import os
import re
from PIL import Image
import io
import numpy as np
import ast
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file (if using)
load_dotenv()

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost:3000",  # Allow your Next.js app origin
    "http://localhost",        # Useful for development if you access directly
    "http://127.0.0.1:3000",  # Alternative localhost
    "*", # DO NOT USE IN PRODUCTION
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Initialize Gemini API (Replace with your API Key)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text from an image using Gemini API
def extract_text_from_image(image_data: bytes) -> str:
    """
    Sends an image to the Gemini API and extracts structured text.
    """
    try:
        img = Image.open(io.BytesIO(image_data))

        # Prompt to ensure structured extraction
        prompt = """
        Extract and structure the prescription details into the following
        fields with clear labels. Use the exact field names as below.
        If a field is missing, use NA. Ensure structured output with one
        field per line, formatted as 'Field: Value'. Keep dosage, duration,
        and frequency in standard format.

        Required Fields:
        Prescription ID
        Prescription Date (Format: YYYY-MM-DD)
        Prescription Status
        Prescription Type
        Patient ID
        Patient Name
        Date of Birth (Format: YYYY-MM-DD)
        Gender
        Allergies
        Medical History
        Medication Name (List if multiple)
        Medication Dosage (Include units, e.g., 500mg, 10ml)
        Medication Form (Tablet, Capsule, Syrup, etc.)
        Frequency (e.g., Twice daily, Once daily)
        Duration (e.g., 7 days, 10 days)
        Quantity
        Refills Allowed
        Instructions (Include any important directions)
        Pharmacy ID
        Pharmacy Name
        Pharmacy Address
        Pharmacy Contact Information
        Diagnosis Code
        Insurance Information
        Signature
        Expiration Date (Format: YYYY-MM-DD)
        Notes

        Provide the extracted text as structured key-value pairs.
        """

        # Use Gemini model
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        response = model.generate_content([prompt, img])

        extracted_text = response.text if response and hasattr(response, "text") else ""
        return extracted_text.strip()

    except Exception as e:
        print(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing error: {e}")


def parse_prescription(text: str) -> Dict:
    """
    Parses the extracted text into structured fields using regex.
    """
    structured_data = {
        "Prescription ID": "NA",
        "Prescription Date": "NA",
        "Prescription Status": "NA",
        "Prescription Type": "NA",
        "Patient ID": "NA",
        "Patient Name": "NA",
        "Date of Birth": "NA",
        "Gender": "NA",
        "Allergies": "NA",
        "Medical History": "NA",
        "Medication Name": [],
        "Medication Dosage": [],
        "Medication Form": "NA",
        "Frequency": "NA",
        "Duration": "NA",
        "Quantity": "NA",
        "Refills Allowed": "NA",
        "Instructions": "NA",
        "Pharmacy ID": "NA",
        "Pharmacy Name": "NA",
        "Pharmacy Address": "NA",
        "Pharmacy Contact Information": "NA",
        "Diagnosis Code": "NA",
        "Insurance Information": "NA",
        "Signature": "NA",
        "Expiration Date": "NA",
        "Notes": "NA",
    }

    # Regex patterns to extract key fields
    patterns = {
        "Prescription ID": r"Prescription ID:\s*(.*)",
        "Prescription Date": r"Prescription Date:\s*(\d{4}-\d{2}-\d{2})",
        "Patient ID": r"Patient ID:\s*(.*)",
        "Patient Name": r"Patient Name:\s*(.*)",
        "Date of Birth": r"Date of Birth:\s*(\d{4}-\d{2}-\d{2})",
        "Medication Name": r"Medication Name:\s*(.*)",
        "Medication Dosage": r"Medication Dosage:\s*(.*)",
        "Instructions": r"Instructions:\s*(.*)",
    }

    for field, pattern in patterns.items():
        matches = re.findall(pattern, text)
        if matches:
            if isinstance(structured_data[field], list):
                structured_data[field].extend(matches)
            else:
                structured_data[field] = matches[0]

    return {k: v for k, v in structured_data.items() if v != "NA" or (isinstance(v, list) and v)}


def clean_list_column(value):
    if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
        try:
            # Convert string to a real list
            value = ast.literal_eval(value)
            if isinstance(value, list):
                return ", ".join(value)  # Convert list to comma-separated string
        except:
            return value  # Return as-is if conversion fails
    return value  # Return original value if not a list


@app.post("/extract_prescriptions/")
async def extract_prescriptions(files: List[UploadFile]):
    """
    Endpoint to upload multiple images, extract prescription data from each,
    and return a list of JSON objects.
    """
    all_results = []
    for file in files:
        try:
            # Validate file type
            if not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type for {file.filename}. Only images are allowed.",
                )

            # Read image data
            image_data = await file.read()

            # Extract text from image
            extracted_text = extract_text_from_image(image_data)

            # Parse the extracted text
            structured_data = parse_prescription(extracted_text)

            # Create a DataFrame
            df = pd.DataFrame([structured_data])

            # Clean the dataframe
            df = df.replace(np.nan, "", regex=True)
            df = df.applymap(clean_list_column)

            # Drop specified columns
            columns_to_drop = ["Prescription Date", "Instructions"]
            df = df.drop(columns=columns_to_drop, errors="ignore")

            # Convert DataFrame to JSON (orient='records' gives a list of dictionaries)
            result_json = df.to_dict(orient="records")[0]  # Extract the dict

            all_results.append(result_json)

        except HTTPException as http_exc:
            # Re-raise HTTPExceptions to avoid being caught by the general
            # exception handler
            all_results.append({"error": http_exc.detail, "filename": file.filename})  # Add error with filename
        except Exception as e:
            print(f"Internal server error processing {file.filename}: {e}")  # Log the error
            all_results.append(
                {"error": f"Internal server error: {str(e)}", "filename": file.filename}
            )  # Add error with filename

    return JSONResponse(content=all_results)
