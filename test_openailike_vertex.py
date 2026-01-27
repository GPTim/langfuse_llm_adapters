import google.oauth2.service_account
import google.auth.transport.requests
import openai
import sys

SERVICE_ACCOUNT_KEY_FILE = "/app/cat/data/tim-cdc-svi-cu00003745p1-l2-c56d1fa6bea7.json"
PROJECT_ID = "tim-cdc-svi-cu00003745p1-l2"  
LOCATION = "global"  
MODEL_NAME = "openai/gpt-oss-120b-maas" #"google/gemini-3-pro-preview"
TEMPERATURE = 1.0

try:
    credentials = google.oauth2.service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_KEY_FILE,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]

   )
    credentials.refresh(google.auth.transport.requests.Request())
    client = openai.OpenAI(
        base_url=f"https://aiplatform.googleapis.com/v1/projects/{PROJECT_ID}/locations/{LOCATION}/endpoints/openapi",
        api_key=credentials.token,
    )
    print("\n--- Running Single Text Generation Test ---")
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Saluta Luca Bincoletto e indica quale LLM sei"}
        ],
        temperature=TEMPERATURE,
    )

    if response.choices:
        print(response.choices[0].message.content)
    else:
        print("Test failed: Model did not return a response.")

except FileNotFoundError:
    print(f"Error: Service account key file not found at '{SERVICE_ACCOUNT_KEY_FILE}'")
    sys.exit(1)
except Exception as e:
    print(f"An unexpected error occurred during setup: {e}")
    sys.exit(1)