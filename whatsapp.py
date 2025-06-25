import os
import requests
from io import BytesIO
from PIL import Image
from typing import Optional
from fastapi import FastAPI, Form, Response
from twilio.rest import Client
from main import full_chain
from dotenv import load_dotenv


conversation_history = {}

load_dotenv()
app = FastAPI()

# Load credentials from environment variables for security
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_SANDBOX_NUMBER = os.getenv("TWILIO_SANDBOX_NUMBER")

twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.post("/webhook")
async def webhook(From: str = Form(...), Body: Optional[str] = Form(None), NumMedia: int = Form(0), MediaUrl0: Optional[str] = Form(None)):
    input_dict = {}
    query = Body or ""
    
    # Get or initialize conversation history for this user
    history = conversation_history.get(From, [])
    history.append({"role": "user", "content": query})
    
    #Image
    if NumMedia > 0 and MediaUrl0:
        response = requests.get(MediaUrl0)
        image = Image.open(BytesIO(response.content))
        input_dict = {"query": query, "image_object": image, "history": history}
    else:
        input_dict = {"query": query, "history": history}

    result = full_chain.invoke(input_dict)
    answer = result.get("result", "Sorry, I could not process your request.")

    # Add assistant response to history
    history.append({"role": "assistant", "content": answer})
    conversation_history[From] = history
    
    twilio_client.messages.create(
        from_=TWILIO_SANDBOX_NUMBER,
        body=answer,
        to=From
    )
    
    return Response(status_code=204)