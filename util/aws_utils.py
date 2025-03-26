import boto3
from dotenv import load_dotenv
import os
from src.setup_log import api_logger, log_file_path
from openai import OpenAI
from sqlalchemy import create_engine, text

load_dotenv()

client = OpenAI()



s3_bucket = os.getenv('S3_BUCKET')
s3_key = os.getenv('S3_KEY')
s3_secret = os.getenv('S3_SECRET')
s3_region = os.getenv('S3_REGION')

# Create an S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=s3_key,
    aws_secret_access_key=s3_secret,
    region_name=s3_region
)



def get_file_keys(doc_ids: list[int]):
    """
    Get the file keys from the database for a list of file IDs.
    """
    try:
        if not doc_ids:
            return []  # Return empty list if no file_ids provided
        
        engine = create_engine(
            f"oracle+oracledb://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_SERVICE')}",
            echo=True
        )

        with engine.connect() as connection:
            # Dynamically generate placeholders for the IN clause
            placeholders = ", ".join([f":file_id_{i}" for i in range(len(doc_ids))])
            query = text(f"""
                SELECT ID_DOCUMENTO,PREFIX_ARQUIVO  
                FROM {os.getenv('DB_SCHEMA')}
                WHERE ID_DOCUMENTO IN ({placeholders})
            """)

            # Create a dictionary of parameters
            params = {f"file_id_{i}": file_id for i, file_id in enumerate(doc_ids)}

            result = connection.execute(query, params)
            file_keys = [{"doc_id": row[0], "prefix": row[1] if row[1] else None} for row in result.fetchall()]  # Extract all file keys

        return file_keys
    except Exception as e:
        logger.error(f"Error {e} com: {doc_ids}")
        return None
        

def build_s3_key(file_key: dict, bucket_folder = 'eprotocolo/'):
    
    s3_key = bucket_folder + file_key
    return s3_key
    

def verify_s3_key(s3_client,file_key, bucket_name=s3_bucket):

    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=file_key)
    if 'Contents' in response:
        for obj in response['Contents']:
            
            return obj['Key']
    else:
        
        
        return 
        
def download_file(file_ids: list[int], s3_client = s3_client, download_path= 'tmp/', bucket_name=s3_bucket):

    try:
        
        file_keys = get_file_keys(file_ids)
    
        
        if file_keys:
            file_paths = []

            # Construct the full local file path
            for file_key in file_keys:
                
                
                
                s3_prefix = file_key.get('prefix')

                if s3_prefix:
                
                    s3_key = build_s3_key(s3_prefix)
                    s3_path = verify_s3_key(s3_client, s3_key)

                    if not s3_path:
                        logger.warning(f"Arquivo não encontrado na S3: {file_key}")
                        continue
                   
                    
                    local_file_path = os.path.join(download_path, s3_path)
                    
                    
                    # Ensure the directory exists
                    local_dir = os.path.dirname(local_file_path)
                    if not os.path.exists(local_dir):
                        os.makedirs(local_dir)

                    s3_client.download_file(bucket_name, s3_path, local_file_path)
                    
                    logger.info(f"File downloaded to: {local_file_path}")
                    file_paths.append(local_file_path)
                else:
                    logger.warning(f'Arquivo Não Encontrado: {file_key}')
            return file_paths
        else:
            return
    except Exception as e:
        logger.error(f"Error downloading files {file_ids}: {e}")
        
        return None

import time

def setup_vector_store(file_paths: list[str] = None, user: str = 'default',):
    """
    Configura o armazenamento de vetores para o assistente.
    """
    if not file_paths:
        return
    try:
        # Create vector store
        vector_store = client.beta.vector_stores.create(
            name=f"{user}_vector_store",
            
        )

        # Prepare file streams
        file_streams = [open(path, "rb") for path in file_paths]

        # Start batch upload
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        # Poll for completion
        upload_status = file_batch.status
        while upload_status != 'completed':
            time.sleep(0.5)  # Wait for a second before polling again
            upload_status = file_batch.status
            if upload_status == 'failed':
                print("File upload failed.")
                return None


        return vector_store.id

    except Exception as e:
        print(f"Error setting up vector store: {e}")
        return None
    finally:
        # Ensure all file streams are closed
        for stream in file_streams:
            stream.close()


def upload_file(file_path):
    """
        Faz upload de um arquivo para processamento pelo assistente.

        - **Request Body**:
            - `file_uploaded` (UploadFile): O arquivo enviado pelo usuário.
        - **Response**:
            - `message` (str): Mensagem de sucesso indicando que o arquivo foi armazenado.
            - `file_id` (str): ID do arquivo armazenado.
    """
    
    if file_path:
        try:
            file_name = os.path.basename(file_path)
            message_file = client.files.create(
                    file=open(file_path,"rb"), purpose="assistants"
                    )

            message_log = message_file.model_dump()
            message_log["filename"] = file_name
            
            resposta = {
                        "file_id": message_file.id,
                        "file_name": file_name
                    }
            logger.info(f"File uploaded: {resposta}")
            return resposta
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return {}


      
    
def add_file_to_user(file_key, user_id,  bucket_name = s3_bucket):
    
    file_name = os.path.basename(file_key)
    user_path = f'tmp/users/{user_id}'
    full_path = f'tmp/users/{user_id}/{file_name}'
    print(full_path)
    try:
        if not os.path.exists(user_path):
            
            os.makedirs(full_path)
            
            s3_client.download_file(bucket_name, file_key, full_path)
            
            return {"status": "successo", "file_path": full_path}
        
        else:
            return {"status": "successo", "message": f"Arquivo ja existe em '{full_path}'","folder": user_path ,"file_path": full_path}
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        return {"status": "error", "message": str(e)}
    
    
    
def extract_file_ids(sync_cursor_page):
    """
    Extract file IDs from a SyncCursorPage[VectorStoreFile] object.
    
    Args:
        sync_cursor_page: The SyncCursorPage object containing VectorStoreFile data.
    
    Returns:
        A list of file IDs (str) extracted from the data.
    """
    try:
        # Extract the 'data' list from the SyncCursorPage object
        file_data = sync_cursor_page.data  # Assuming `sync_cursor_page` has a `data` attribute

        # Extract 'id' from each VectorStoreFile in the 'data' list
        file_ids = [file.id for file in file_data]

        return file_ids
    except Exception as e:
        print(f"Error extracting file IDs: {e}")
        return []