import os
import streamlit as st
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.messages import AIMessage, HumanMessage
# Define the proper prompt template for the agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_react_agent
from PIL import Image
from sqlalchemy import create_engine
import pandas as pd
from datetime import datetime
import settings as s
import re

# ----------------------
# Configuration & Setup
# ----------------------
def initialize_session_state():
    """Initialize all required session state variables"""
    defaults = {
        "messages": [AIMessage(content="Hello! How can I assist you today?")],
        "customer_name": None,
        "customer_phone": None,
        "customer_id": None,
        "order_id": None,
        "order_products": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def create_snowflake_connection():
    """Create and verify Snowflake database connection"""
    try:
        snowflake_user = st.secrets["credentials"]["snowflake_user"]
        snowflake_password = st.secrets["credentials"]["snowflake_password"]
        snowflake_account = st.secrets["credentials"]["snowflake_account"]

        # Build the connection URI (using f-strings to insert actual credentials)
        connection_uri = (
            f"snowflake://{snowflake_user}:{snowflake_password}@{snowflake_account}/"
            "RETURN_REQUEST_DB/RETURN_REQUEST_SCHEMA?warehouse=COMPUTE_WH&role=ACCOUNTADMIN"
        )
        engine = create_engine(connection_uri)
        return SQLDatabase(engine=engine), engine
    except Exception as e:
        st.error(f"❌ Failed to connect to Snowflake: {e}")
        st.stop()

# ----------------------
# Core Business Logic
# ----------------------
def validate_phone_number(phone: str, engine) -> tuple:
    """Validate customer phone number and return customer details"""
    clean_phone = re.sub(r'\D', '', phone)
    try:
        query = f"""
            SELECT CUSTOMER_ID, CUSTOMER_NAME 
            FROM CUSTOMER 
            WHERE MOBILE_NUMBER = '{clean_phone}'
        """
        df = pd.read_sql_query(query, engine)
        return (df['customer_id'].iloc[0], df['customer_name'].iloc[0]) if not df.empty else (None, None)
    except Exception as e:
        raise RuntimeError(f"Phone validation error: {e}")

def get_order_products(order_id: str, customer_id: str, engine) -> dict:
    """Retrieve products associated with an order"""
    try:
        query = f"""
            SELECT OI.ORDER_ITEM_ID, P.PRODUCT_NAME 
            FROM ORDERS O
            JOIN ORDER_ITEMS OI ON O.ORDER_ID = OI.ORDER_ID
            JOIN PRODUCTS P ON OI.PRODUCT_ID = P.PRODUCT_ID
            WHERE O.ORDER_ID = {order_id} 
            AND O.CUSTOMER_ID = {customer_id}
        """
        df = pd.read_sql_query(query, engine)
        return df.set_index('order_item_id')['product_name'].to_dict()
    except Exception as e:
        raise RuntimeError(f"Order products error: {e}")

def check_return_eligibility(product_id: int, engine) -> dict:
    """Check return eligibility for a specific product"""
    try:
        query = f"""
            SELECT RP.IS_RETURNABLE, RP.RETURN_POLICY_DETAILS,
                   RP.NO_OF_DAYS, O.ORDER_DELIVERED_DATE
            FROM ORDER_ITEMS OI
            JOIN ORDERS O ON OI.ORDER_ID = O.ORDER_ID
            JOIN PRODUCTS P ON OI.PRODUCT_ID = P.PRODUCT_ID
            JOIN RETURN_POLICY RP ON P.CATEGORY = RP.CATEGORY
            WHERE OI.ORDER_ITEM_ID = {product_id}
        """
        df = pd.read_sql_query(query, engine)
        return df.iloc[0].to_dict() if not df.empty else None
    except Exception as e:
        raise RuntimeError(f"Return eligibility error: {e}")

# ----------------------
# Product Identification
# ----------------------
def normalize_product_input(user_input: str, products: dict) -> int:
    """Identify product using LLM and fuzzy matching"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    try:
        # Generate product list for LLM context
        products_list = "\n".join([f"{pid}: {name}" for pid, name in products.items()])
        
        # LLM-based identification
        prompt = ChatPromptTemplate.from_template(
            """Identify the product from this input. Return ONLY the ProductID as integer.
            Available products:
            {products}
            Input: {input}"""
        )
        chain = prompt | st.session_state.llm | StrOutputParser()
        llm_output = chain.invoke({"products": products_list, "input": user_input}).strip()
        
        # Try numeric conversion first
        try:
            return int(llm_output)
        except ValueError:
            # Fuzzy matching fallback
            clean_input = re.sub(r'[^a-z0-9]', '', user_input.lower())
            for pid, name in products.items():
                clean_name = re.sub(r'[^a-z0-9]', '', name.lower())
                if clean_input == clean_name:
                    return pid
            return None
    except Exception as e:
        raise RuntimeError(f"Product identification error: {e}")


def main():
    # Initialize application
    # Streamlit page configuration
    jade_image = Image.open("company_logo/jadeglobalsmall.png")
    initialize_session_state()
    st.set_page_config(page_title="AI-Powered Sales Bot | Jadeglobal", page_icon=jade_image)
    st.title("🚀 Return Assistant Bot")
    st.subheader("Your Smart Assistant for Sales Success")
    
    db, engine = create_snowflake_connection()
    openai_key = st.secrets["api"]["OpenAI_Secret_Key"]
    st.session_state.llm = ChatOpenAI(temperature=0, model_name="gpt-4o-2024-11-20", api_key=openai_key)

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg.type.capitalize()):
            st.write(msg.content)

    # Process new input
    if user_input := st.chat_input("Your message:"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("User"):
            st.write(user_input)

        response = None

        try:
            # ----------------------
            # Conversation Flow Logic
            # ----------------------
            # 1. Handle personal information queries first
            key_list = ["policy", "product", "order", "address", "delivery", "mobile","DOB","birth"]
            if st.session_state.customer_id and any(keyword in user_input.lower() for keyword in key_list):
                toolkit = SQLDatabaseToolkit(db=db, llm=st.session_state.llm)
                
                prompt = ChatPromptTemplate.from_template(
                    """You are a helpful assistant that can answer questions about customer orders and return policies.
                    Use the provided tools to look up information and provide accurate responses.
                    You are a ReAct agent. Follow **exactly** this format:
                    Thought: <your internal reasoning>
                    Action: <tool name> {tool_names}
                    Action Input: <tool input> {tools}
                    ...
                    Thought: <your final internal reasoning>{agent_scratchpad}
                    Final Answer: <answer to the user>

                    Do not add anything extra. Do not use bullet points. Do not say "Observation:" or "Explanation:". 
                    The user’s question is: {input}

                    Customer Information:
                    - Name: {customer_name}
                    - Phone: {customer_phone}
                """
                )
                
                # Create the agent
                agent = create_react_agent(
                    llm=st.session_state.llm,
                    tools=toolkit.get_tools(),
                    prompt=prompt
                )
                
                # Create the AgentExecutor with increased iteration and time limits
                agent_executor = AgentExecutor(
                    agent=agent,
                    tools=toolkit.get_tools(),
                    verbose=True,  # Set to True for debugging
                    handle_parsing_errors=True,  # Handle errors gracefully
                    max_iterations=20,  # Increase iteration limit
                    max_execution_time=60  # Increase time limit (in seconds)
                )
                
                # Invoke the agent with the user input and customer information
                try:
                    result = agent_executor.invoke({
                        "input": user_input,
                        "customer_name": st.session_state.customer_name,
                        "customer_phone": st.session_state.customer_phone
                    })
                    
                    # Check if the agent returned a valid response
                    if "output" in result and result["output"]:
                        response = result["output"]
                    else:
                        # Fallback response if the agent fails to provide a valid output
                        response = "I couldn't find the requested information. Please try again or provide more details."
                except Exception as e:
                    # Handle any errors that occur during agent execution
                    response = f"⚠️ Error processing your request: {str(e)}"

            # 2. Return initiation flow
            elif "return" or "hi" in user_input.lower():
                if not st.session_state.customer_phone:
                    response = "Sure! May I have your phone number?"
                elif not st.session_state.order_id:
                    response = "Please provide your Order ID"
                elif not st.session_state.order_products:
                    response = "Which product would you like to return?"

            # 3. Phone number validation
            elif not st.session_state.customer_phone and re.search(r'\d', user_input):
                customer_id, customer_name = validate_phone_number(user_input, engine)
                if customer_id:
                    st.session_state.customer_id = customer_id
                    st.session_state.customer_name = customer_name
                    st.session_state.customer_phone = re.sub(r'\D', '', user_input)
                    response = f"Hi {customer_name}! Please provide your Order ID."
                else:
                    response = "❌ Phone number not found in our system."

            # 4. Order ID handling
            elif not st.session_state.order_id and user_input.isdigit():
                products = get_order_products(user_input, st.session_state.customer_id, engine)
                if products:
                    st.session_state.order_id = user_input
                    st.session_state.order_products = products
                    product_list = "\n".join([f"ProductID {pid}: {name}" for pid, name in products.items()])
                    response = f"""Found order {user_input}!
                    Available products:\n{product_list}\nWhich product would you like to return?"""
                else:
                    response = "❌ Order not found or doesn't belong to you."

            # 5. Product identification and return check
            elif st.session_state.order_products:
                product_id = normalize_product_input(user_input, st.session_state.order_products)
                if product_id:
                    policy = check_return_eligibility(product_id, engine)
                    if policy:
                        delivered_date = pd.to_datetime(policy["order_delivered_date"]).date()
                        days_since = (datetime.now().date() - delivered_date).days
                        
                        if policy["is_returnable"] == "Yes" and days_since <= policy["no_of_days"]:
                            response = f"""✅ {st.session_state.order_products[product_id]} is returnable!
                            Delivered: {delivered_date} ({days_since} days ago)"""
                        else:
                            reason = "category restrictions" if policy["is_returnable"] != "Yes" \
                                      else f"return window expired ({policy['no_of_days']} days)"
                            response = f"""❌ Cannot return {st.session_state.order_products[product_id]}
                            Reason: {reason} | Delivered: {delivered_date}"""
                    else:
                        response = "⚠️ No return policy found for this product"
                else:
                    response = "🔍 I'm sorry, Please Entered correct query"

            # 6. Fallback for unrecognized input
            else:
                response = "I'm sorry, I don't understand. How can I assist you today?"

        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

        # Append and display response
        if response:
            st.session_state.messages.append(AIMessage(content=response))
            with st.chat_message("Assistant"):
                st.write(response)

if __name__ == "__main__":
    main()