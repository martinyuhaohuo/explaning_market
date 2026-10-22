import os
from dotenv import load_dotenv
from langfuse import get_client
from langfuse.langchain import CallbackHandler
from langchain_core.callbacks import BaseCallbackHandler

load_dotenv()
_LANGFUSE_KEYS = (
    "LANGFUSE_HOST",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
)

if all(os.environ.get(key) for key in _LANGFUSE_KEYS):
    langfuse = get_client()
    langfuse_handler = CallbackHandler()
else:
    # Keep callers' callback lists valid while making tracing optional.
    langfuse = None
    langfuse_handler = BaseCallbackHandler()
