import os
import sys
import logging
import tempfile
import warnings
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential
from azure.core.exceptions import ResourceNotFoundError
from urllib.parse import quote_plus
from dotenv import load_dotenv

# --- Global Initialization Flags ---
INITIALIZATION_COMPLETE = False

# --- Private Credential Store ---
# Secrets are loaded at runtime from Azure Key Vault or environment variables.
# Never hard-code or commit credential values.
_cfg = {}

# --- Logger Setup ---
config_logger = logging.getLogger("APP_CONFIG_STATUS")
config_logger.propagate = False
if not config_logger.handlers:
    config_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    config_logger.addHandler(handler)

# Suppress noisy Azure logs
logging.getLogger("azure").setLevel(logging.ERROR)

# --- Logging ---
# Single unified logger for the entire application.
# force=True ensures this config takes effect even when gunicorn/uvicorn
# sets up logging first. stream=sys.stdout ensures logs appear in
# Azure App Service Log Stream and docker/container stdout.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout,
    force=True
)
logger = logging.getLogger("TicketAgent")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", message=".*n_jobs value 1 overridden to 1.*")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*force_all_finite.*")
warnings.filterwarnings("ignore", category=UserWarning, module="umap")


# --- Configuration ---
load_dotenv()

# --- SECRET RETRIEVAL (Key Vault + Fallback) ---
KEY_VAULT_URL = os.environ.get("AZURE_KEY_VAULT_URL")

# Use configurable secret names for your own Key Vault.
AZURE_OPENAI_KEY_SECRET_NAME = os.getenv(
    "AZURE_OPENAI_KEY_SECRET_NAME", "AZURE-OPENAI-API-KEY"
)
POSTGRES_PASSWORD_SECRET_NAME = os.getenv(
    "POSTGRES_PASSWORD_SECRET_NAME", "POSTGRES-DATABASE-PASSWORD"
)
APP_API_KEY_SECRET_NAME = os.getenv(
    "APP_API_KEY_SECRET_NAME", "APP-API-KEY"
)
SESSION_ENCRYPTION_SECRET_NAME = os.getenv(
    "SESSION_ENCRYPTION_SECRET_NAME", "SESSION-ENCRYPTION-KEY"
)

if not _cfg:
    _azure_openai_key = None
    _postgres_conn_string = None
    _api_key_secret_kv = None   # NEW: Initialize Key Vault variable
    _session_enc_key_kv = None  # NEW: Initialize Key Vault variable
    _secrets_found = False

    # 1. Try Azure Key Vault
    if KEY_VAULT_URL:
        try:
            credential = DefaultAzureCredential()
            key_vault_client = SecretClient(vault_url=KEY_VAULT_URL, credential=credential)

            _azure_openai_key = get_secret(AZURE_OPENAI_KEY_SECRET_NAME)
            _db_pass_raw = get_secret(POSTGRES_PASSWORD_SECRET_NAME)
            _api_key_secret_kv = get_secret(APP_API_KEY_SECRET_NAME)
            _session_enc_key_kv = get_secret(SESSION_ENCRYPTION_SECRET_NAME)

            _azure_openai_key = get_secret(os.getenv("AZURE_OPENAI_KEY_SECRET_NAME", "AZURE-OPENAI-API-KEY"))
            _db_pass_raw = get_secret(os.getenv("POSTGRES_PASSWORD_SECRET_NAME", "POSTGRES-DATABASE-PASSWORD"))
            
            # --- NEW ADDITIONS START ---
            _api_key_secret_kv = get_secret(os.getenv("APP_API_KEY_SECRET_NAME", "APP-API-KEY"))
            _session_enc_key_kv = get_secret(os.getenv("SESSION_ENCRYPTION_SECRET_NAME", "SESSION-ENCRYPTION-KEY"))
            # --- NEW ADDITIONS END ---

            _db_host = os.environ.get("PGHOST")
            _db_user = os.environ.get("PGUSER")
            _db_name = os.environ.get("PGDATABASE")
            _db_port = os.environ.get("PGPORT", "5432") 

            if _db_pass_raw and _db_host and _db_user and _db_name:
                safe_password = quote_plus(_db_pass_raw)
                _postgres_conn_string = (
                    f"postgresql+psycopg://{_db_user}:{safe_password}@{_db_host}:{_db_port}/{_db_name}"
                )
                config_logger.info("Constructed DB connection string from KV password + Env vars.")
            else:
                config_logger.warning("Key Vault password found, but missing PGHOST/USER/DB in .env")

            if _azure_openai_key and _postgres_conn_string:
                _secrets_found = True
                config_logger.info("Core credentials secured via AZURE KEY VAULT.")

        except Exception as e:
            config_logger.error(f"Failed to connect to Key Vault: {e}")

    # 2. Fallback to Environment Variables (Local .env)
    config_logger.warning("Falling back to local environment variables.")

    _azure_openai_key = _azure_openai_key or os.environ.get("AZURE_OPENAI_API_KEY")
    _postgres_conn_string = _postgres_conn_string or os.environ.get("DATABASE_URL")

    _cfg['aoai'] = _azure_openai_key
    _cfg['pg'] = _postgres_conn_string
    
    # --- MODIFIED: Prioritize KV variables, fallback to os.environ ---
    _cfg['auth'] = _api_key_secret_kv if _api_key_secret_kv else os.environ.get("API_KEY_SECRET")
    _cfg['enc'] = _session_enc_key_kv if _session_enc_key_kv else os.environ.get("SESSION_ENCRYPTION_KEY")

# API Security
API_KEY_NAME = "X-API-Key"
# Storage Paths — Use persistent disk on Azure, temp dir locally
# Azure App Service provides /home as persistent storage (250GB disk)
# /tmp on Azure is in-memory (tmpfs) and consumes RAM!
if os.path.isdir("/home"):
    APP_DATA_DIR = "/home/app_data"
    config_logger.info("Azure detected. Using /home/app_data for persistent storage.")
else:
    APP_DATA_DIR = os.path.join(tempfile.gettempdir(), "ticket_agent_data")
    config_logger.info(f"Local environment. Using {APP_DATA_DIR} for storage.")

UPLOAD_DIR = os.path.join(APP_DATA_DIR, "uploads")
SQLITE_DIR = os.path.join(APP_DATA_DIR, "sqlite_data")
DATA_RETENTION_DAYS = 15
MAX_SESSIONS_PER_USER = int(os.environ.get("MAX_SESSIONS_PER_USER", 2))

# Create directories if they don't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(SQLITE_DIR, exist_ok=True)

# Azure Config
AZURE_DEPLOYMENT = os.environ["AZURE_OPENAI_DEPLOYMENT"]
AZURE_MINI_DEPLOYMENT=os.environ["AZURE_OPENAI_MINI_DEPLOYMENT"]
AZURE_EMBEDDING_DEPLOYMENT = os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]
AZURE_API_VERSION = os.environ["AZURE_OPENAI_API_VERSION"]
AZURE_ENDPOINT = os.environ["AZURE_OPENAI_ENDPOINT"]

# DB Config
DB_SCHEMA = os.environ.get("PG_SCHEMA")
DB_TABLE = os.environ.get("PG_TABLE")

ROW_LIMIT = 20

# --- Credential Accessor Functions ---
# Secrets are loaded at runtime from Azure Key Vault or environment variables.
# Never hard-code or commit credential values.

def get_azure_api_key():
    """Returns the Azure OpenAI API key."""
    return _cfg.get('aoai')

def get_pg_connection_string():
    """Returns the PostgreSQL connection string."""
    return _cfg.get('pg')

def get_auth_secret():
    """Returns the API authentication secret."""
    return _cfg.get('auth')

def get_encryption_key():
    """Returns the session encryption key."""
    return _cfg.get('enc')