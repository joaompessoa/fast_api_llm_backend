from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from sqlalchemy import create_engine
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.mongodb import MongoDBSaver
from dotenv import load_dotenv
from src.custom_prompt import CUSTOM_PROMPT
import os
from pymongo import MongoClient 
from langchain_ollama import ChatOllama

load_dotenv()

builder = StateGraph(MessagesState)
mongodb_client = MongoClient(os.getenv('MONGO_URI'))
memory = MongoDBSaver(mongodb_client)


def build_agent(
        model: str,
        db_uri: str, 
        api_key = os.getenv('OPENAI_API_KEY'),
        verbose = True,
        sample_rows = 10,
        base_prompt = CUSTOM_PROMPT,
        memory = memory,
        db_type = 'PostgreSQL',
        base_url = None,
        temperature = 0.2
    ):

    system_message = CUSTOM_PROMPT.format(dialect=db_type, top_k=sample_rows)

    engine = create_engine(db_uri)

    db = SQLDatabase(
        engine=engine,
        view_support=True,
        include_tables=tables,
        sample_rows_in_table_info=sample_rows,
    )
    if not base_url:
           

        llm = ChatOpenAI(
            api_key=api_key,
            temperature=temperature,
            verbose=verbose,
            model=model,
            base_url = base_url
        
        )
    else:
        llm = ChatOllama(
            model=model,
            temperature=temperature,
            verbose=verbose,
            num_ctx=16000,
            base_url=base_url
        )


    graph = create_react_agent(
        llm,
        tools=SQLDatabaseToolkit(db=db, llm=llm).get_tools(),
        state_modifier=system_message,
        checkpointer=memory,
        debug=True
    )
    

    return graph