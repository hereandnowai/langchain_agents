# The updated import to use the latest langchain-ollama package
from langchain_ollama.chat_models import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
import yfinance as yf
from typing import Dict
import json

@tool
def get_stock_prices(tickers: str) -> str:
    """
    Fetches the current stock price and currency for a comma-separated string of ticker symbols.
    For example, to get the price for Google and Infosys, the input should be 'GOOGL,INFY.NS'.
    Returns a JSON string where each ticker maps to a dictionary containing 'price' and 'currency'.
    """
    # Sanitize the input by converting to uppercase and stripping whitespace
    ticker_list = [ticker.strip().upper() for ticker in tickers.split(',')]
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
                prices[ticker] = {'price': round(float(price), 2), 'currency': currency}
            else:
                prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
        except Exception as e:
            print(f"Warning: Could not find data for {ticker}. Error: {e}")
            prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
            
    # Return a simple JSON string for the agent to easily parse
    return json.dumps(prices)

def run_finance_agent_stock_price():
    """
    Creates and runs an agent that uses the local llama3.1:8b model via Ollama.
    """
    # ------------------- This is the key change -------------------
    # Point to your local Ollama instance with the specific Llama 3.1 model
    llm = ChatOllama(model="llama3.1:8b", temperature=0)
    # --------------------------------------------------------------
    
    tools = [get_stock_prices]

    # Use the standard ReAct prompt, which works very well with instruction-tuned models
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor with robust error handling
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True
    )

    # Get user input for stock tickers
    user_input = input("Please enter the stock tickers you want to look up, separated by commas (e.g., GOOGL,INFY.NS): ")

    # Provide a clear, detailed prompt to the agent
    input_prompt = (
        f"What are the current stock prices for {user_input}? "
        "The tool will return a JSON string with price and currency (e.g., 'USD' or 'INR') for each stock. "
        "Once you have this information from the tool, you MUST immediately provide the final answer. "
        "Do not use the tool more than once per ticker. "
        "Format the final answer using '₹' for 'INR' and '$' for 'USD'. "
        "For example: 'The current stock price for INFY.NS is ₹1550.75.'"
    )

    # Invoke the agent with the user's request
    result = agent_executor.invoke({"input": input_prompt})
    
    print("\nFinal Answer from Agent:")
    print(result.get('output'))

if __name__ == "__main__":
    run_finance_agent_stock_price()