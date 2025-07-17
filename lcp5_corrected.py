from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

@tool
def get_stock_prices(tickers: str) -> Dict[str, Dict[str, Any]]:
    """
    Fetches the current stock price and currency for a comma-separated string of ticker symbols.
    For example, to get the price for Google and Infosys, the input should be 'GOOG,INFY.NS'.
    Returns a dictionary where each ticker maps to another dictionary containing 'price' and 'currency'.
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
                prices[ticker] = {'price': price, 'currency': currency}
            else:
                # Fallback for tickers that don't have fast_info
                hist = stock.history(period='1d')
                if not hist.empty:
                    prices[ticker] = {
                        'price': hist['Close'].iloc[-1],
                        'currency': info.get('currency', 'N/A') # Try to get currency anyway
                    }
                else:
                    prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
        except Exception as e:
            print(f"Warning: Could not find data for {ticker}. It may be an invalid ticker. Error: {e}")
            prices[ticker] = {'price': 0.00, 'currency': 'N/A'}
            
    return prices

def run_finance_agent_stock_price():
    """
    Creates and runs an agent that can use the get_stock_prices tool.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
    )
    
    tools = [get_stock_prices]

    # Pull the recommended react prompt from the hub.
    prompt = hub.pull("hwchase17/react")

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Get user input for stock tickers
    user_input = input("Please enter the stock tickers you want to look up, separated by commas (e.g., GOOG,INFY.NS): ")

    # Construct a detailed input string for the agent.
    # We now instruct it on how to handle the new output format from our tool.
    input_prompt = (
        f"What are the current stock prices for {user_input}? "
        "The tool will return the price and the currency (e.g., 'USD' or 'INR') for each stock. "
        "Your final answer should format the price with the correct currency symbol. "
        "Use '$' for 'USD' and '₹' for 'INR'. For example: "
        "'The current stock price for [TICKER] is ₹[PRICE].' or "
        "'The current stock price for [TICKER] is $[PRICE].' "
        "List each stock on a new line."
    )

    # Invoke the agent. You only need to pass the 'input' variable.
    result = agent_executor.invoke({"input": input_prompt})
    
    print("\nFinal Answer from Agent:")
    print(result.get('output'))

if __name__ == "__main__":
    run_finance_agent_stock_price()