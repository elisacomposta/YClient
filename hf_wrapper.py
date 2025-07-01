from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
import os

app = FastAPI()

HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
        user_input = data["messages"][-1]["content"]
        HF_API_KEY = os.getenv("HF_API_KEY")
        
        payload = {"inputs": user_input}
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        
        # Chiamata API a Hugging Face
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        
        # Verifica se la risposta è corretta
        if response.status_code == 200:
            result = response.json()
            if "generated_text" in result[0]:
                return {"choices": [{"message": {"role": "assistant", "content": result[0]["generated_text"]}}]}
            else:
                return JSONResponse(status_code=500, content={"error": "No generated_text in response"})
        else:
            return JSONResponse(status_code=response.status_code, content={"error": "Hugging Face API call failed", "details": response.text})
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
