from pymongo import MongoClient
from src.agent_setup import build_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import time
from datetime import datetime
from fastapi import File, UploadFile, HTTPException
from pathlib import Path
from openai import OpenAI
import asyncio
from PyPDF2 import PdfReader
import docx
import os
import httpx
import json
from ollama import chat
import psutil
from util.logger_setup import logger
import requests
from ollama.client import Client


MONGO_URI = os.getenv('MONGO_URI')

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client.db
collection = db.collection
logs = db.logs
file_logs = db.file_logs
run_logs = db.run_logs
thread_logs = db.thread_logs
message_logs = db.message_logs


async def get_messages(thread_id):
        
    client = MongoClient(MONGO_URI)
    db = client.tcmpa
    col = db.chat_pdf

    message_history = col.find_one({"thread_id": thread_id}, {"_id": 0, "conversations": 1})
    client.close()

    return message_history.get("conversations", []) if message_history else []



async def push_logs_mongo(type: str, logs: dict):
    """
    Pushes logs to MongoDB
    """
    logger.info(f"Pushing logs to MongoDB with the {type}: {logs}")
    
    if type == "file":
        file_logs.insert_one(logs)
    elif type == "run":
        run_logs.insert_one(logs)
    elif type == "message":
        message_logs.update_one(
            {'thread_id': logs.get('thread_id')},
            {'$set': logs},
            upsert=True
        )
    elif type == "thread":
        thread_logs.insert_one(
            logs
        )
    pass

async def process_log_messages(thread_id, run_id):
    messages = list(client.beta.threads.messages.list(thread_id=thread_id, run_id=run_id))
    
    for message in messages:
        logger.info(f"Processing message for logging: {message}")
        asyncio.create_task(
            push_logs_mongo(
                type='message',
                logs=message.model_dump(exclude_none=True)
            )
        )


client = OpenAI()
# Database URI
database_uri = os.getenv('DB_URI')



async def process_query_with_langchain(agent_instance,query: str, thread_id: int = 1):
    log = {
                "human_message": query,
                "run_query": None,
                "ai_answer": None,
                "ai_details": [],
                "tool_details": [],
                "model": None,
                "input_tokens": 0,
                "output_tokens": 0,
    }

    config = {"configurable": {"thread_id": thread_id}}
    logger.info(f"Processing query: {query} with the following config: {config}")

    if hasattr(agent_instance, "memory"):
        messages = agent_instance.memory.load_memory_variables()
        logger.info(f"Memory for {thread_id}: {messages}")
    else:
        logger.info(f"No memory found for {thread_id}")

    # Process the query using LangChain
    start_time = time.time()
    logger.info(f"Processing query: {query}")
    events = agent_instance.stream(
                    {"messages": [("user", query)]},
                    config=config,
                    stream_mode="values",
                    debug=True
                )
    message = ''
    for event in events:
        message = event["messages"][-1]
        print(message)
       
        answer = message.content
        if type(message) == AIMessage:

            run_tool = message.tool_calls
            if run_tool:
                run_query = run_tool[-1].get("args", {})
                log["run_query"] = run_query.get("query", {})
                ai_details = {

                        "id": message.id,            
                        "response_metadata": message.response_metadata,
                }
               
                log["ai_details"].append(ai_details)
        elif type(message) == ToolMessage:
                        
            tool_details = {
                        "id": message.id,
                        "tool_used": message.name,
                        "tool_output": message.content,
                }
            log["tool_details"].append(tool_details)
    end_time = time.time() - start_time
    log["input_tokens"] = message.usage_metadata.get("input_tokens")
    log["output_tokens"] = message.usage_metadata.get("output_tokens")
    log["model"] = message.response_metadata.get("model_name", {})
    log["ai_answer"] = answer
    log["execution_time(s)"] = round(end_time, 2)
    log["entry_time"] = datetime.now()

    logger.info(f"Mongo Entry: {log}")

    logs.insert_one(log)
    return answer


async def upload_file(file_uploaded: UploadFile):
    """
    Uploads a file to the assistant's file storage
    Args:
        file_uploaded (UploadFile): File to be uploaded
    Returns:
        dict: File uploaded successfully
    """
    try:
        message_file = client.files.create(
            file=open(file_uploaded.filename, "rb"),
            purpose="assistants"
        )

        return {"message": "File uploaded successfully", "file_id": message_file.id}, 200
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        

def process_query_with_openai_assistant():
    pass

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(file_path)
    for page in reader.pages:
        text += page.extract_text() + "\n"
        text = text.replace("\n", " ")
    return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    text = text.replace("\n", " ")
    return text

def get_file_content(file_path: list[str]):
    """
    Reads the content of a PDF or DOCX file and adds it to the context window
    of a language model using the OpenAI API.

    Parameters:
        file_path (str): Path to the PDF or DOCX file.
        model (str): Model name (e.g., 'gpt-4').
        api_key (str): Your OpenAI API key.
        max_chunk_size (int): Maximum token size for each API call.
    """

    file_content = ""

    for file in file_path:
        # Determine file type and extract text
        try:
            if file.endswith(".pdf"):
                file_name = os.path.basename(file)
                content = extract_text_from_pdf(file)
                file_content += f'/n/n === {file_name} === /n/n {content}'
            elif file.endswith(".docx"):
                file_name = os.path.basename(file)
                content = extract_text_from_docx(file)
                file_content += f'/n/n === {file_name} === /n/n {content}'
            else:
                raise ValueError("Unsupported file type. Please provide a PDF or DOCX file.")
        except ValueError as e:
            logger.error(f'Erro tentando tirar o conteudo de um arquivo')
    return file_content
    
 
def process_chat_ollama(messages ,url = "http://ollama:11434", model = "llama3.2"):
    """
    Process a chat query using the OLLAMA API.
    """
    client = Client(
        host=url
    )
    payload = client.chat(
        model=model,
        messages=messages,
        stream=False,
        options={
            "temperature": 0.3,
            "num_ctx": 4096
        }
    )
    
    logger.info(f"Processing chat with OLLAMA: {payload}")
    
    ai_message =payload.message
    return ai_message
   
    

def benchmark_chat_ollama(messages, url="http://ollama:11434/api/chat", model="llama3.2", context_size = 4096):
    """
    Process a chat query using the OLLAMA API and collect benchmark metrics, including system usage.
    """
    payload = {
        "messages": messages,
        "model": model,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": context_size
        }
    }

    col = db.benchmark_llm
    
    logger.info(f"Benchmark chat with OLLAMA: {payload}")
    
    # Capture system metrics before the request
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
    
    start_time = time.time()
    response = requests.post(url, json=payload)
    end_time = time.time()
    
    # Capture system metrics after the request
    cpu_after = psutil.cpu_percent(interval=None)
    memory_after = psutil.virtual_memory().used / (1024 ** 3)  # Convert to GB
    
    if response.status_code == 200:
        data = response.json()
        ai_message = data.get("message", {})
        
        # Extract benchmarking metrics
        metrics = {
            "model": data.get("model"),
            "payload":payload,
            "model_answer": ai_message,
            "total_duration": data.get("total_duration", 0) / 1e9,
            "load_duration": data.get("load_duration", 0) / 1e9,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration": data.get("prompt_eval_duration", 0) / 1e9,
            "eval_count": data.get("eval_count", 0),
            "eval_duration": data.get("eval_duration", 0) / 1e9,
            "done_reason": data.get("done_reason"),
            "actual_request_time": end_time - start_time,  # Convert to seconds
            "cpu_usage_before": cpu_before,
            "cpu_usage_after": cpu_after,
            "memory_usage_before": memory_before,
            "memory_usage_after": memory_after
        }

        col.insert_one(metrics)
        
        logger.info(f"Benchmark results: {metrics}")
        return ai_message
    
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")
    
