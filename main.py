# Import Libraries
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from utils import chatbot, load_LLM_groq, Load_vectordb, save_vectordb, Update_VectorDB
from langchain.embeddings import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from utils import chatbot, load_LLM_groq, Load_vectordb

# Load environment variables
_ = load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Initialize embeddings and vector database
embeddings = HuggingFaceEmbeddings()

try:
    vectordb = Load_vectordb("VectorDB", embeddings)
except Exception as e:
    raise RuntimeError(f"Failed to load vector database: {e}")

# Initialize the LLM agent
try:
    LLMAgent = load_LLM_groq(vectordb, "llama-3.1-70b-versatile")
except Exception as e:
    raise RuntimeError(f"Failed to initialize LLM: {e}")

# Initialize FastAPI app
app = FastAPI(debug=True)

# Templates setup
templates = Jinja2Templates(directory="templates")

# Define a Pydantic model for chatbot requests
class ChatbotRequest(BaseModel):
    user_prompt: str


@app.get("/", response_class=HTMLResponse)
async def serve_html(request: Request):
    """
    Serve the HTML interface at the root URL.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chatbot")
async def chatbot_endpoint(request: ChatbotRequest):
    """
    Chatbot endpoint to process user prompts.
    """
    try:
        response = chatbot(LLM=LLMAgent, user_prompt=request.user_prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

# Run the FastAPI app on a custom host and port
if __name__ == "__main__":
    uvicorn.run(app)
