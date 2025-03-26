from fastapi import FastAPI, HTTPException, Request, UploadFile, Response
from openai import AssistantEventHandler
from typing_extensions import override
from fastapi.middleware.cors import CORSMiddleware
from src.agent_setup import build_agent
from src.process_chat import *
from util.aws_utils import download_file, add_file_to_user, upload_file, setup_vector_store, extract_file_ids
from src.custom_prompt import CHAT_PDF_PROMPT
from src.custom_prompt_en import query_system_prompt
from util.logger_setup import logger
import random
from dotenv import load_dotenv
import os
import json
import asyncio
import openai
from fastapi.responses import StreamingResponse
import requests

load_dotenv()

client = openai.OpenAI()

MONGO_URI = os.getenv("MONGO_URI")



# Initialize FastAPI app
app = FastAPI(
    title="LLM Backend",
    description="API para integração de um backend de modelos de linguagem",
    version="0.1-beta",
    debug=True
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



def setup_query(data: dict, messages: list[dict]):
    
    files = data.get("files")
    query = data.get("query")

    files_content = ""
    file_names = ""
    
    try:
            
        if files:
            print(files)
        
            file_path = download_file(file_ids=files)
                
            if file_path:
                files_content = get_file_content(file_path)
                file_names = [os.path.basename(path) for path in file_path]

            else:
                files_content = f"Não foi possível extrair o conteúdo dos arquivos Enviados"
        if not query:
            query = {
                    "role": "user",
                    "content": f'Faça um resumo dos arquivos, o nome dos arquivos sao {file_names} segue o conteudo dos arquivos: {files_content}. Informe o usuário se não foi possível extrar os arquivos'
                }
                
            messages.append(query)
                
                
        else:
            query = {
                    "role": "user",
                    "content": query + f'\n Você  recebeu os arquivos {file_names} anteriormente que o usuário enviou para dar contexto a conversa. Segue o conteudo {files_content}' if files else query
                }
                
            messages.append(query)
    except HTTPException:
        return 

    return messages
    
            
# Routes
@app.get("/ia_api/health", tags=["Health Check"])
async def health_check():
    """
    Verifica o estado da API.
    Retorna uma mensagem indicando que a API está funcionando corretamente.

    - **Status 200**: API está saudável.
    """
    try:
        return {"status": "OK", "message": "API is healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ia_api/resposta", tags=['Respostas'], responses={200: {"description": "Resposta do modelo de linguagem"}})
async def answer(request: Request):
    """
    Processa a pergunta do usuário e retorna uma resposta.

    - **Request Body**:
        - `query` (str): A pergunta ou consulta enviada pelo usuário.
        - `thread_id` (int, opcional): ID da thread para manter o contexto. Se ausente, será gerado um ID aleatório.
    - **Response**:
        - `answer` (str): A resposta gerada pelo modelo de linguagem.
    Parameters:
        request (Request): Requisição HTTP recebida.
    
    """
    try:
        data = await request.json()
        logger.info(f"Data received {data}")
        query = data.get("query")
        #sem resposta erro
        if not query:
            raise HTTPException(status_code=400, detail="Pergunta vazia!")

        thread_id = data.get("thread_id", random.getrandbits(16))

        openai_agent = build_agent(
            model="gpt-4o",
            verbose=True,
            db_uri=os.getenv("DB_URI"),
        )

        answer = await process_query_with_langchain(query=query, thread_id=thread_id, agent_instance=openai_agent)
        return {"answer": answer, "thread_id": thread_id}
    except Exception as e:
        logger.error(f"Error Processing the Request with {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Um erro ocorreu ao processar sua requisição")


@app.post("/ia_api/ia_local", tags=['Respostas'])
async def ia_local(request: Request):
    """
    Processa a pergunta do usuário com uma IA local.

    - **Request Body**:
        - `query` (str): A pergunta ou consulta enviada pelo usuário.
        - `thread_id` (int, opcional): ID da thread para manter o contexto. Se ausente, será gerado um ID aleatório.
    - **Response**:
        - `answer` (str): A resposta gerada pelo modelo local.
    """
    print('Entrou no endpoint')
    try:
        data = await request.json()
        logger.info(f"Data received for local: {data}")

        query = data.get("query")
        if not query:
            raise HTTPException(status_code=400, detail="Pergunta vazia!")

        thread_id = data.get("thread_id", random.getrandbits(16))
        local_agent =  build_agent(
            model="qwen2.5-coder",
            api_key="ollama",
            verbose=True,
            db_uri=os.getenv("DB_URI"),
            base_prompt=query_system_prompt,
            base_url='http://ollama:11434' # Setup com DNS do docker-compose
            
        )


            
        answer =  process_query_with_langchain(agent_instance=local_agent,query=query, thread_id=thread_id)
        return Response(
            content=json.dumps(
                {"answer": await answer, 
                 "thread_id": thread_id
                 },
                indent=4,
                default=str,
            ),
            media_type="application/json",
            status_code=200,
        )
    except Exception as e:
        logger.error("Error Processing the Request", exc_info=True)
        return HTTPException(status_code=500, detail="Um erro ocorreu ao processar sua requisição")



@app.post("/ia_api/chat_pdf/resposta_openai", tags=['Respostas'])
async def answer_pdf(request: Request):
    """
        Processa o conteúdo de um arquivo PDF para gerar uma resposta.

        - **Request Body**:
            - `query` (str): Pergunta baseada no conteúdo do arquivo PDF.
            - `thread_id` (str, opcional): ID da thread para manter o contexto.
            - `file_id` (str, opcional): ID do arquivo previamente enviado.
        - **Response**:
            - `answer` (str): Resposta gerada com base no conteúdo do PDF e na pergunta.
            - `thread_id` (str): ID da thread utilizada ou criada.
    """
    data = await request.json()
    logger.info(f"Data received for PDF: {data}")
    
    query = data.get("query", None)
    thread_id = data.get("thread_id", None)
    vector_id = data.get("vector_id", None)
    files = data.get("files")
    
    if files:
    
        file_path = download_file(file_ids=files)
        
        if file_path:
            vector_id = setup_vector_store(file_path)
            file_names = [os.path.basename(file) for file in file_path]
            logger.info(f"Vector store created: {vector_id} with the files: {file_names}")
        
        else:
            raise HTTPException(
                    status_code=404, detail=f"Arquivo solicitado não foi encontrado na base de dados"
                        )
           
    try:
        
        if not thread_id:
            thread = client.beta.threads.create()
            thread_id = thread.id
            logger.info(f"Thread created: {thread_id}")

        attachments = []
        if vector_id:
            client.beta.threads.update(
                thread_id=thread_id,
                tool_resources={"file_search": 
                    {"vector_store_ids": [vector_id]}},
                )
            vector_store_name = client.beta.vector_stores.retrieve(vector_id).name
            vector_files = client.beta.vector_stores.files.list(vector_id, filter='completed')
            file_ids =  extract_file_ids(vector_files)
            
            
            if not query:
                query = f'Faça um resumo dos arquivos na lista {file_names} citando seu nome e o conteudo'
            else:
                query = query + f'\n Voce tem acesso aos arquivos {file_names} que o usuario enviou para contexto da conversa'
                
            message = client.beta.threads.messages.create(
                thread_id=thread_id,
                role="user",
                content= query,
                attachments=[
                    {"file_id": file_id, "tools": [{"type": 'code_interpreter'}, {"type": 'file_search'}]
                    } for file_id in file_ids
                ] if file_ids else []
                
            )
        else:
            message = client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=query
            )
        logger.info(f"Message created: {message.model_dump()}")

        

        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=os.getenv("ASSISTANT_ID"),
            include=["step_details.tool_calls[*].file_search.results[*].content"],
        )

       

        asyncio.create_task(
            push_logs_mongo(type="run", logs=run.model_dump(exclude_none=True))
        )

        messages = list(
            client.beta.threads.messages.list(thread_id=thread_id, run_id=run.id)
        )
        logger.info(f"Messages: {messages}")
        asyncio.create_task(process_log_messages(thread_id, run.id))

        message_content = messages[0].content[0].text
        
        struct_answer = json.dumps(
                {"answer": message_content.value, "thread_id": thread_id, "vector_id": vector_id},
                indent=4,
                default=str,
            )
      
        return Response(
            content=struct_answer,
            media_type="application/json",
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao processar a mensagem"
        )

import uuid

@app.post("/ia_api/chat_pdf/resposta", tags=['Respostas'])
async def answer_pdf_local(request: Request):
    """
        Processa o conteúdo de um arquivo PDF para gerar uma resposta.

        - **Request Body**:
            - `query` (str): Pergunta baseada no conteúdo do arquivo PDF.
            - `messages` (list, opcional): Historico da conversa se houver.
            - `files` (list, opcional): ID do arquivo enviado para download na S3.
        - **Response**:
            - `answer` (str): Resposta gerada com base no conteúdo do PDF e na pergunta.
            - `messages` (list): Historico da conversa
    """
    data = await request.json()
    logger.info(f"Solicitação recebida com os dados: \n\n {data}")

    client = MongoClient(MONGO_URI)
    db = client.tcmpa
    col = db.chat_pdf


             
    try:
        
        thread_id = data.get('thread_id') or str(uuid.uuid4())
        messages_history = await get_messages(thread_id)
        messages_history.insert(0,{'role': 'system','content':CHAT_PDF_PROMPT})
        messages = setup_query(data=data,messages=messages_history)

        resposta = process_chat_ollama(messages=messages,model='llama3.2')

        message_content = resposta.content
        message_role = resposta.role

        messages.append({'role': message_role, 'content': message_content})

        col.update_one({"thread_id": thread_id}, {"$set": {"conversations": messages}}, upsert=True)
        
  
        return Response(
            content=json.dumps(
                {"answer": message_content, 
                 "thread_id": thread_id
                 },
                indent=4,
                default=str,
            ),
            media_type="application/json",
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao processar a mensagem"
        )
    


@app.post("/ia_api/benchmark", tags=['Respostas'])
async def benchmark_local_llm(request: Request):
    data = await request.json()
    logger.info(f"Solicitação recebida com os dados: \n\n {data}")

    client = MongoClient(MONGO_URI)
    db = client.benchmark
    col = db.chats

    model = data.get('model')
    ctx_size = data.get('ctx_size') if data.get('ctx_size') else 4096
    
    if not model:
        model = 'deepseek-r1'
             
    try:
        
        thread_id = data.get('thread_id') or str(uuid.uuid4())
        messages_history = await get_messages(thread_id)
        messages = setup_query(data=data,messages=messages_history)

        resposta = benchmark_chat_ollama(messages=messages,model=model, context_size=ctx_size)

        message_content = resposta.get("content")

        messages.append(resposta)

        col.update_one({"thread_id": thread_id}, {"$set": {"conversations": messages}}, upsert=True)
        
  
        return Response(
            content=json.dumps(
                {"answer": message_content, 
                 "thread_id": thread_id
                 },
                indent=4,
                default=str,
            ),
            media_type="application/json",
            status_code=200,
        )
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ocorreu um erro ao processar a mensagem"
        )
    