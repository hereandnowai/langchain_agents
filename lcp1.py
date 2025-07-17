from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()

def run_hello_langchain():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
    response = llm.invoke("Tell me about HERE AND NOW AI - Artificial Intelligence Research Institute")
    print(response.content)

if __name__ == "__main__":
    run_hello_langchain()