import os
import requests
from io import BytesIO
from PIL import Image
from typing import Optional
from fastapi import FastAPI, Form, Response
from twilio.rest import Client
from main import full_chain_with_memory
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# Load credentials from environment variables for security
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SANDBOX_NUMBER = os.getenv("TWILIO_SANDBOX_NUMBER")
MAX_TWILIO_BODY_LENGTH = 1600

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.post("/webhook")
async def webhook(From:str = Form(...), Body: Optional[str] = Form(None), NumMedia: int = Form(0), MediaUrl0: Optional[str] = Form(None)):
    input_dict = {}
    query = Body or ""

    #Image
    if NumMedia > 0 and MediaUrl0:
        response = requests.get(MediaUrl0)
        image = Image.open(BytesIO(response.content))
        input_dict = {"query": query, "image_object": image}
    else:
        input_dict = {"query": query}

    session_id = From  # Use phone number as session ID
    result = full_chain_with_memory.invoke(
        input_dict,
        config={"configurable": {"session_id": session_id, "k": 5}}
    )

    answer = result.get("result", "Sorry, I could not process your request.")
    chunks = [answer[i:i+MAX_TWILIO_BODY_LENGTH] for i in range(0, len(answer), MAX_TWILIO_BODY_LENGTH)]

    for chunk in chunks:
        twilio_client.messages.create(
            from_=TWILIO_SANDBOX_NUMBER,
            body=chunk,
            to=From
        )
    return Response(status_code=204)