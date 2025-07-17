from langchain_google_genai import ChatGoogleGenerativeAI # For using Google's generative AI models
from langchain_core.prompts import ChatPromptTemplate # For creating prompts for the AI model
from langchain_core.output_parsers import StrOutputParser # For getting a simple text output from the AI
from dotenv import load_dotenv # To load environment variables from a .env file
import os # To interact with the operating system, like getting environment variables

load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

def run_text_summarizer():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=google_api_key)

    prompt_template = "Summarize the following text:\n\n{text}\n\nSummary:"
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = prompt | llm | StrOutputParser()

    with open("profile-of-hereandnowai.txt", "r") as f:
        long_text = f.read()

    summary = chain.invoke({"text": long_text})
    print(summary)

if __name__ == "__main__":
    run_text_summarizer()