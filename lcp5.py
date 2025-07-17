from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import tool
from dotenv import load_dotenv
import os
import yfinance as yf
from typing import Dict
import ast
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file
load_dotenv()
google_api_key = os.getenv("GEMINI_API_KEY")

@tool
def get_stock_prices(tickers: str) -> Dict[str, float]:
    """
    Fetches the current stock price for a comma-separated string of ticker symbols.
    For example, to get the price for Google and Infosys, the input should be 'GOOG,INFY.NS'.
    """
    # Split the comma-separated string into a list of tickers
    ticker_list = [ticker.strip() for ticker in tickers.split(',')]
    
    prices = {}
    for ticker in ticker_list:
        if not ticker:  # Skip any empty strings that might result from splitting
            continue
        
        stock = yf.Ticker(ticker)
        # Use 'fast_info' for a quicker lookup of the last price
        price = stock.fast_info.get('last_price')
        
        if price:
            prices[ticker] = price
        else:
            # If fast_info fails, try history as a fallback
            todays_data = stock.history(period='1d')
            if not todays_data.empty:
                prices[ticker] = todays_data['Close'].iloc[-1]
            else:
                print(f"Warning: Could not find price for {ticker}. It may be an invalid ticker.")
                prices[ticker] = 0.00
    return prices

def run_finance_agent_stock_price():
    """
    Creates and runs an agent that can use the get_stock_prices tool.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=google_api_key,
        # Removed the 'stop' argument as it was causing warnings and not strictly needed here.
    )
    
    tools = [get_stock_prices]

    # Define the ChatPromptTemplate with all necessary input variables and placeholders
    prompt = ChatPromptTemplate(
        input_variables=["input", "agent_scratchpad", "tool_names", "tools"], # Added "tools" to input_variables
        messages=[
            ("system", "You are a helpful assistant that can fetch stock prices."),
            ("system", "You have access to the following tools: {tool_names}. Use them to answer the user's question."),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
            (
                "system",
                "When you have retrieved the stock prices, provide the final answer in the format: 'The current stock price for [TICKER] is $[PRICE].' for each stock. If there are multiple stocks, list them all. Always provide a final answer after using the tool."
            ),
        ]
    )

    # Create the agent
    agent = create_react_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # Get user input for stock tickers
    user_input = input("Please enter the stock tickers you want to look up, separated by commas (e.g., GOOG,INFY.NS): ")

    # Invoke the agent with the user's input
    # Pass 'tools' and 'tool_names' explicitly as required by the prompt's input_variables
    result = agent_executor.invoke({"input": f"What are the current stock prices for {user_input}?", 
                                   "tools": tools, 
                                   "tool_names": ", ".join([tool.name for tool in tools])})
    
    print("\nFinal Answer from Agent:")
    print(result.get('output'))

if __name__ == "__main__":
    run_finance_agent_stock_price()
