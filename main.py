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
from typing import List, Dict, Tuple
from dotenv import load_dotenv
import uuid

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

from pydantic import BaseModel
from typing import List, Optional, Dict, Any

from simple_salesforce import Salesforce
import snowflake.connector

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Import for Chrome service
from selenium.webdriver.chrome.service import Service
from fastapi.staticfiles import StaticFiles
# from webdriver_manager.chrome import ChromeDriverManager

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

GOOGLE_API_KEY_V2= os.environ.get("GOOGLE_API_KEY_V2")
genai.configure(api_key=GOOGLE_API_KEY_V2)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# --- Salesforce Configuration ---
SALESFORCE_USERNAME = "godcares.ndubuisi.btech2021878@agentforce.com"
SALESFORCE_PASSWORD = "good@1234"
SALESFORCE_SECURITY_TOKEN = "tMaMITpwIoMwpuRU9Gk9tz2a"
SALESFORCE_SOQL_QUERY = "SELECT Patient_Name__c, Medication_Name__c, Instructions__c, Medication_Dosage__c FROM Symbi_Pharmacy__c"


# --- Snowflake Configuration ---
SNOWFLAKE_USER = "GODCARES"
SNOWFLAKE_PASSWORD = "meowCrack it@1011"
SNOWFLAKE_ACCOUNT = "UOKKATU-VK33181"
SNOWFLAKE_SQL_QUERY = "SELECT * FROM COMPANY_PHARMACY.DETAIL.PRESCRIPTIONS;"


# --- Data Models ---
class QueryInput(BaseModel):
    user_query: str


# --- Salesforce Integration ---
def connect_salesforce():
    try:
        sf = Salesforce(
            username=SALESFORCE_USERNAME,
            password=SALESFORCE_PASSWORD,
            security_token=SALESFORCE_SECURITY_TOKEN,
        )
        return sf
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Salesforce Authentication Error: {e}"
        )


def query_salesforce(sf: Salesforce) -> List[Dict[str, Any]]:
    """Execute a SOQL query and return results as a list of dictionaries."""
    try:
        result = sf.query_all(SALESFORCE_SOQL_QUERY)
        records = result.get("records", [])
        return records  # Return the records directly as a list of dicts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Salesforce Query Error: {e}")


@app.post("/salesforce/")
async def get_salesforce_data(query_input: QueryInput):
    """
    Fetches data from Salesforce based on a predefined SOQL query and processes
    it using Gemini, based on the user's input query.
    """
    sf = connect_salesforce()
    data = query_salesforce(sf)

    if not data:
        raise HTTPException(
            status_code=404, detail="No data retrieved from Salesforce."
        )

    formatted_data = "\n".join([str(item) for item in data])

    final_prompt = f"""
    {query_input.user_query}

    Respond ONLY with the exact answer. STRICTLY DO NOT INCLUDE ANYTHING ELSE.
    Be direct and to the point. NO extra words, NO emojis, NO irrelevant text.

    If the answer requires a structured response, use numbering or bullet points. Otherwise, respond in a concise paragraph.

    If the question CANNOT be answered based on the provided data, respond ONLY with:
    'I cannot answer this question because the required data is not available in the Salesforce database.'

    Data for Analysis:
    {formatted_data}
    """

    try:
        response = model.generate_content(final_prompt)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error communicating with Gemini: {e}"
        )


# --- Snowflake Integration ---
def connect_snowflake():
    try:
        conn = snowflake.connector.connect(
            user=SNOWFLAKE_USER,
            password=SNOWFLAKE_PASSWORD,
            account=SNOWFLAKE_ACCOUNT,
        )
        return conn
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Snowflake Connection Error: {e}"
        )


def query_snowflake(conn: snowflake.connector.SnowflakeConnection) -> List[Dict[str, Any]]:
    """Execute a SQL query and return results as a list of dictionaries."""
    try:
        with conn.cursor() as cur:
            cur.execute(SNOWFLAKE_SQL_QUERY)
            columns = [desc[0] for desc in cur.description]  # Fetch column names
            rows = cur.fetchall()

            # Convert each row into a dictionary
            results = []
            for row in rows:
                row_dict = dict(zip(columns, row))
                results.append(row_dict)  # Append dictionaries directly

            return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Snowflake Query Error: {e}")
    finally:
        cur.close()  # Ensure cursor is closed after execution

@app.post("/snowflake/")
async def get_snowflake_data(query_input: QueryInput):
    """
    Fetches data from Snowflake based on a predefined SQL query and processes it using Gemini, based on the user's input query.
    """
    conn = connect_snowflake()
    try:
        data = query_snowflake(conn)

        if not data:
            raise HTTPException(
                status_code=404, detail="No data retrieved from Snowflake."
            )

        formatted_data = "\n".join([str(item) for item in data])

        final_prompt = f"""
        {query_input.user_query}

        Respond ONLY with the exact answer. STRICTLY DO NOT INCLUDE ANYTHING ELSE.
        Be direct and to the point. NO extra words, NO emojis, NO irrelevant text.

        If the question CANNOT be answered based on the provided data, respond ONLY with:
        'I cannot answer this question because the required data is not available in the snowflake database.'

        Data for Analysis:
        {formatted_data}
        """

        try:
            response = model.generate_content(final_prompt)
            return {"response": response.text}
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error communicating with Gemini: {e}"
            )
    finally:
        conn.close()

# --- Upload Salesforce API ---

class Credentials(BaseModel):
    username: str
    password: str
    task_name: str

def run_part_1_snowflake(username, password):
    try:
        # Set up the Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--headless')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(120)

        driver.set_window_size(1920, 1080)
        wait = WebDriverWait(driver, 60)

        driver.get("https://dm-us.informaticacloud.com/identity-service/home")

        username_div = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        username_input = username_div.find_element(By.TAG_NAME, "input")
        username_input.clear()
        username_input.send_keys(username)

        password_div = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        password_input = password_div.find_element(By.TAG_NAME, "input")
        password_input.clear()
        password_input.send_keys(password)

        login_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".submitBtn.loginBtn.ui-infa-button-emphasized button"))
        )

        try:
            login_button.click()
            wait.until(EC.visibility_of_element_located((By.ID, "root")))
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=400, detail="Incorrect Username or Password")

        fi_task_url = "https://usw5.dm-us.informaticacloud.com/diUI/products/integrationDesign/main/fiTask/3wXTuKQq6DmcgOVgxPdU4l/edit"
        driver.get(fi_task_url)

        time.sleep(200)

        try:
            run_button = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(@class, 'd-button--call-to-action')]//span[text()='Run']/.."
                ))
            )

            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", run_button)
            driver.execute_script("arguments[0].click();", run_button)

        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=500, detail=f"Error interacting with Run button: {str(e)}")

        finally:
            driver.quit()

        return {"status": "Success"}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Part 1 failed: {str(e)}")

def run_part_1_salesforce(username, password):
    try:
        # Set up the Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--headless')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(120)

        driver.set_window_size(1920, 1080)
        wait = WebDriverWait(driver, 60)

        driver.get("https://dm-us.informaticacloud.com/identity-service/home")

        username_div = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        username_input = username_div.find_element(By.TAG_NAME, "input")
        username_input.clear()
        username_input.send_keys(username)

        password_div = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        password_input = password_div.find_element(By.TAG_NAME, "input")
        password_input.clear()
        password_input.send_keys(password)

        login_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".submitBtn.loginBtn.ui-infa-button-emphasized button"))
        )

        try:
            login_button.click()
            wait.until(EC.visibility_of_element_located((By.ID, "root")))
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=400, detail="Incorrect Username or Password")

        fi_task_url = "https://usw5.dm-us.informaticacloud.com/diUI/products/integrationDesign/main/mapping/6LisynXT7sggmHH7EWvl0L/edit"
        driver.get(fi_task_url)

        time.sleep(200)

        try:
            run_button = wait.until(
                EC.element_to_be_clickable((
                    By.XPATH,
                    "//button[contains(@class, 'infaButton') and contains(@class, 'infaButton-1') and .//span[text()='Run']]"
                ))
            )

            driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", run_button)
            time.sleep(5)
            driver.execute_script("arguments[0].click();", run_button)
            time.sleep(5)

        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=500, detail=f"Error interacting with Run button: {str(e)}")

        finally:
            driver.quit()

        return {"status": "Success"}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Part 1 failed: {str(e)}")


def run_part_2(username, password, task_name):
    try:
        # Set up the Selenium WebDriver
        options = webdriver.ChromeOptions()
        options.add_argument('--no-sandbox')
        options.add_argument('--headless')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-gpu')
        
        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(120)

        driver.set_window_size(1920, 1080)
        wait = WebDriverWait(driver, 60)

        driver.get("https://dm-us.informaticacloud.com/identity-service/home")

        username_div = wait.until(EC.visibility_of_element_located((By.ID, "username")))
        username_input = username_div.find_element(By.TAG_NAME, "input")
        username_input.clear()
        username_input.send_keys(username)

        password_div = wait.until(EC.visibility_of_element_located((By.ID, "password")))
        password_input = password_div.find_element(By.TAG_NAME, "input")
        password_input.clear()
        password_input.send_keys(password)

        login_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, ".submitBtn.loginBtn.ui-infa-button-emphasized button"))
        )

        try:
            login_button.click()
            wait.until(EC.visibility_of_element_located((By.ID, "root")))
        except Exception as e:
            driver.quit()
            raise HTTPException(status_code=400, detail="Incorrect Username or Password")

        monitoring_url = "https://usw5.dm-us.informaticacloud.com/diUI/products/integrationDesign/main/MonitorJobs"
        status_found = False
        start_time = time.time()
        timeout = 1800
        poll_interval = 15
        task_xpath = f"//span[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{task_name.lower()}')]"

        print("Started while")
        while (time.time() - start_time) < timeout and not status_found:
            try:
                driver.get(monitoring_url)

                task_element = wait.until(
                    EC.visibility_of_element_located((By.XPATH, task_xpath))
                )

                row = task_element.find_element(By.XPATH, "./ancestor::div[contains(@role, 'row')]")
                status_element = row.find_element(
                    By.XPATH, ".//img[contains(@class, 'job-status-icon')]/following-sibling::span"
                )
                current_status = status_element.text.strip().lower()

                if current_status in ["success", "failed"]:
                    status_found = True
                    break
                else:
                    time.sleep(poll_interval)

            except Exception as e:
                time.sleep(poll_interval)
                continue

        driver.quit()

        if not status_found:
            return {"final_status": "Timeout: Status check exceeded 30 minutes"}
        else:
            return {"final_status": current_status}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        driver.quit()
        raise HTTPException(status_code=500, detail=f"Part 2 failed: {str(e)}")


@app.post("/upload_to_snowflake")
async def upload_to_snowflake(credentials: Credentials):
    try:
        part1_result = run_part_1_snowflake(credentials.username, credentials.password)
        if part1_result["status"] == "Success":
            print("Part 1 success")
            # part2_result = run_part_2(credentials.username, credentials.password, credentials.task_name)
            return {"part1_status": "Success"}
        else:
            raise HTTPException(status_code=500, detail="Part 1 failed")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_to_salesforce")
async def upload_to_salesforce(credentials: Credentials):
    try:
        part1_result = run_part_1_salesforce(credentials.username, credentials.password)
        
        if part1_result["status"] == "Success":
            print("Part 1 success")
            # part2_result = run_part_2(credentials.username, credentials.password, credentials.task_name)
            return {"part1_status": "Success"}
        else:
            raise HTTPException(status_code=500, detail="Part 1 failed")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
# --- Plot Generation ---
# Create a directory for static images
IMAGE_SAVE_DIR = "static/graphs"
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

genai.configure(api_key=GOOGLE_API_KEY_V2)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Snowflake connection parameters
def get_snowflake_connection():
    return snowflake.connector.connect(
        user="GODCARES",
        password="meowCrack it@1011", 
        account="UOKKATU-VK33181"
    )

def plain_query_snowflake(query):
    """Execute a SQL query and return results."""
    conn = get_snowflake_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            return cur.fetchall()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database query error: {e}")
    finally:
        conn.close()

@app.get("/healthcare-analysis")
async def generate_healthcare_graphs():
    # Queries
    queries = [
        "SELECT * FROM COMPANY_PHARMACY.DETAIL.PRESCRIPTIONS;",
        "SELECT * FROM COMPANY_PHARMACY.DETAIL.HOSPITAL_BEDS;",
        "SELECT * FROM COMPANY_PHARMACY.DETAIL.MEDICATIONS;"
    ]
    
    # Fetch data
    data_sets = []
    for query in queries:
        data_sets.append(plain_query_snowflake(query))
    
    # Prepare data for LLM
    formatted_data = ["\n".join([str(row) for row in dataset]) for dataset in data_sets]
    
    # Generate questions
    prompt_questions = f"""
    IMAGINE YOU ARE A DATA ANALYST WITH EXPERTISE IN HEALTHCARE DATA ANALYTICS.

    Based on the provided data, generate **6 important analytical questions** that:
    - Can be answered using the given data
    - Have definitive numerical or categorical answers
    - Can be visualized with a graph using matplotlib

    STRICT RULES:
    1. ONLY output **six** questions, each on a **new line**.
    2. Do NOT include explanations, context, or unnecessary text. Just the questions.

    Data for analysis:
    1. Prescription Data: {formatted_data[0]}
    2. Hospital Bed Data: {formatted_data[1]}
    3. Medication Inventory Data: {formatted_data[2]}
    """

    # Generate questions
    response_questions = model.generate_content(prompt_questions)
    questions = response_questions.text.strip().split("\n")
    
    # Store graph details
    graph_details = []
    
    # Generate graphs for each question
    print(len(questions))
    for idx, question in enumerate(questions, 1):
        prompt_code = f"""
        Given the following healthcare dataset, generate a Python script using **matplotlib** to visualize the answer to the question below.

        QUESTION: "{question}"

        **Rules:**
        - Output only **valid executable Python code** (No explanations, no markdown, just the raw code).
        - Output code must not be wrapped in ``` ```.
        - Output code should only be to represent only one matplotlib graph.
        - Should not have a header name of Python.. Give output code such that if i copied the entire output it should run on my collab notbook.
        - Use the given data directly and ensure code is written correctly without any errors.
        - Ensure that every Matplotlib graph code includes a line to save the image using plt.savefig('name_of_graph').
        - Ensure the script generates a meaningful and readable visualization.

        Data:
        1. Prescription Data: {formatted_data[0]}
        2. Hospital Bed Data: {formatted_data[1]}
        3. Medication Inventory Data: {formatted_data[2]}
        """

        # Generate graph code
        response_code = model.generate_content(prompt_code)
        generated_code = response_code.text.strip()

        # Create a local namespace for execution
        local_namespace = {
            'plt': plt, 
            'prescription_data': data_sets[0],
            'hospital_bed_data': data_sets[1],
            'medication_data': data_sets[2]
        }

        # Execute the code in a controlled environment
        try:
            exec(generated_code, {}, local_namespace)
            
            # Generate unique filename
            filename = f"graph_{idx}_{uuid.uuid4().hex}.png"
            filepath = os.path.join(IMAGE_SAVE_DIR, filename)
            
            # Save plot
            plt.tight_layout()
            plt.savefig(filepath)
            plt.close()  # Close the plot to free memory
            
            # Store graph details
            graph_details.append({
                'question': question,
                'image_path': f"/static/graphs/{filename}"
            })
        except Exception as e:
            print(f"Error generating graph: {e}")
    
    return JSONResponse(content={"graphs": graph_details})