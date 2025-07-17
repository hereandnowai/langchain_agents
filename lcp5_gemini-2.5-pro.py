from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import Dict, Any
import json

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

@tool
def get_stock_prices(tickers: str) -> str:
    """
    Fetches the current stock price and currency for a comma-separated string of ticker symbols.
    For example, to get the price for Google and Infosys, the input should be 'GOOG,INFY.NS'.
    Returns a JSON string where each ticker maps to a dictionary containing 'price' and 'currency'.
    """
    ticker_list = [ticker.strip() for ticker in tickers.split(',')]
    prices = {}
    for ticker in ticker_list:
        if not ticker:
            continue
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.fast_info
            
            price = info.get('last_price')
            currency = info.get('currency')

            if price and currency:
                # IMPORTANT: Round the price and convert to a standard float.
                # This creates a cleaner output for the agent to parse.
                prices[ticker] = {'price': round(float(price), 2), 'currency': currency}
            else:
                prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
        except Exception as e:
            print(f"Warning: Could not find data for {ticker}. Error: {e}")
            prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
            
    # IMPORTANT: Return a simple JSON string instead of a complex dictionary object.
    # This is much easier for the LLM to handle reliably.
    return json.dumps(prices)

def run_finance_agent_stock_price():
    """
    Creates and runs an agent that can use the get_stock_prices tool.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
    )
    
    tools = [get_stock_prices]

    prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(llm, tools, prompt)
    
    # Add handle_parsing_errors=True for more robust error handling.
    # This helps the agent recover if it generates a malformed response.
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    user_input = input("Please enter the stock tickers you want to look up, separated by commas (e.g., GOOG,INFY.NS): ")

    # Refined prompt to be more direct about the final step.
    input_prompt = (
        f"What are the current stock prices for {user_input}? "
        "The tool will return a JSON string with price and currency (e.g., 'USD' or 'INR') for each stock. "
        "Once you have this information from the tool, you MUST immediately provide the final answer. "
        "Do not use the tool more than once. "
        "Format the final answer using '₹' for 'INR' and '$' for 'USD'. "
        "For example: 'The current stock price for INFY.NS is ₹1550.75.'"
    )

    result = agent_executor.invoke({"input": input_prompt})
    
    print("\nFinal Answer from Agent:")
    print(result.get('output'))

if __name__ == "__main__":
    run_finance_agent_stock_price()