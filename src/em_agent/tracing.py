import os
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()
for k in ('LANGFUSE_HOST', 'LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'):
    assert os.environ.get(k), f'missing {k} in .env'
print('env OK')

# Initialize Langfuse client
langfuse = get_client()
# Initialize Langfuse CallbackHandler for Langchain (tracing)
langfuse_handler = CallbackHandler()