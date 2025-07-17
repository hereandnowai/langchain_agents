from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

def simple_chatbot():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key)
    print("Caramel AI - Chatbot of HERE AND NOW AI")
    print("Ask anything you like or type quit to exit")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            break
        response = llm.invoke(user_input)
        print(f"Bot: {response.content}")
    
simple_chatbot()