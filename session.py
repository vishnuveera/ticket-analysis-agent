import os
import glob
import pickle
import hashlib
import logging
from typing import List, Dict, Optional, Any
from sqlalchemy import create_engine, text, inspect as sql_inspect
from cryptography.fernet import Fernet
from config import DB_SCHEMA, DB_TABLE, get_encryption_key, SQLITE_DIR, UPLOAD_DIR, MAX_SESSIONS_PER_USER, logger
from models import AgentState


def hash_email(email: str) -> str:
    """Creates a deterministic SHA-256 hash of the lowercased, trimmed email."""
    return hashlib.sha256(email.strip().lower().encode('utf-8')).hexdigest()

class SessionManager:
    def __init__(self, connection_string: str, schema: str, table: str):
        self.engine = create_engine(
            connection_string,
            pool_size=1, max_overflow=2, pool_recycle=1800
        )
        self.full_table_name = f"{schema}.{table}"
        key = get_encryption_key()
        if not key:
            raise ValueError("SESSION_ENCRYPTION_KEY not found in environment variables.")
        self.cipher = Fernet(key.encode() if isinstance(key, str) else key)


    def save_session(self, session_id: str, state: dict, title: Optional[str] = None, user_email_hash: Optional[str] = None):
        """Pickles and saves state to Postgres."""
        state_copy = state.copy()
        state_copy['db_engine'] = None 
        
        blob = pickle.dumps(state_copy)
        blob = self.cipher.encrypt(blob)

        query = text(f"""
            INSERT INTO {self.full_table_name} (session_id, session_blob, session_title, user_email_hash, updated_at)
            VALUES (:sid, :blob, :title, :email_hash, NOW())
            ON CONFLICT (session_id) 
            DO UPDATE SET 
                session_blob = :blob, 
                session_title = COALESCE(:title, {self.full_table_name}.session_title),
                user_email_hash = COALESCE(:email_hash, {self.full_table_name}.user_email_hash),
                updated_at = NOW();
        """)
        with self.engine.begin() as conn:
            conn.execute(query, {"sid": session_id, "blob": blob, "title": title, "email_hash": user_email_hash})

    def list_sessions(self, user_email_hash: str) -> List[Dict]:
        """Fetches metadata for sessions belonging to a specific user."""
        query = text(f"""
            SELECT session_id, session_title, updated_at 
            FROM {self.full_table_name} 
            WHERE user_email_hash = :email_hash
            ORDER BY updated_at DESC
            LIMIT :max_sessions
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"email_hash": user_email_hash, "max_sessions": MAX_SESSIONS_PER_USER}).fetchall()
            
        return [
            {
                "session_id": row[0],
                "title": row[1] if row[1] else f"Untitled Session ({str(row[0])[:8]})",
                "updated_at": row[2].isoformat() if row[2] else ""
            }
            for row in result
        ]

    def load_session(self, session_id: str) -> Optional[dict]:
        """Loads state from Postgres."""
        query = text(f"SELECT session_blob FROM {self.full_table_name} WHERE session_id = :sid")
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": session_id}).fetchone()
        if result:
            try:
                # Decrypt the blob from DB before unpickling
                raw_data = bytes(result[0])
                decrypted_blob = self.cipher.decrypt(raw_data)
                return pickle.loads(decrypted_blob)
            except Exception as e:
                # if the key is wrong.
                logger.error(f"Decryption failed: {e}")
                return None
        return None

    def verify_ownership(self, session_id: str, user_email_hash: str) -> bool:
        """Checks if a session belongs to the given user."""
        query = text(f"""
            SELECT 1 FROM {self.full_table_name} 
            WHERE session_id = :sid AND user_email_hash = :email_hash
        """)
        with self.engine.connect() as conn:
            result = conn.execute(query, {"sid": session_id, "email_hash": user_email_hash}).fetchone()
        return result is not None
    
    def update_session_title(self, session_id: str, new_title: str, user_email_hash: str):
        """Updates the title in both the SQL column AND the state blob (keeps them in sync)."""
        # Load the state blob so we can update session_title inside it
        state = self.load_session(session_id)
        if state:
            state["session_title"] = new_title
            # save_session updates BOTH the blob and the SQL column in one operation
            self.save_session(session_id, state, title=new_title, user_email_hash=user_email_hash)
        else:
            # Fallback: if blob can't be loaded, at least update the SQL column
            query = text(f"""
                UPDATE {self.full_table_name} 
                SET session_title = :title 
                WHERE session_id = :sid AND user_email_hash = :email_hash
            """)
            with self.engine.begin() as conn:
                conn.execute(query, {"title": new_title, "sid": session_id, "email_hash": user_email_hash})

    def _delete_session_internal(self, session_id: str):
        """Deletes a session row and associated files. No ownership check — caller must verify."""
        query = text(f"DELETE FROM {self.full_table_name} WHERE session_id = :sid")
        with self.engine.begin() as conn:
            conn.execute(query, {"sid": session_id})
        
        # Clean up associated files
        try:
            sqlite_path = os.path.join(SQLITE_DIR, f"{session_id}.db")
            if os.path.exists(sqlite_path):
                os.remove(sqlite_path)
                logger.info(f"Deleted SQLite file: {sqlite_path}")
            
            # Upload files are named {session_id}_{filename}
            for f in glob.glob(os.path.join(UPLOAD_DIR, f"{session_id}_*")):
                os.remove(f)
                logger.info(f"Deleted upload file: {f}")
        except Exception as e:
            logger.warning(f"File cleanup failed for session {session_id}: {e}")

    def delete_session(self, session_id: str, user_email_hash: str):
        """Deletes a session row and associated files (with ownership check)."""
        if not self.verify_ownership(session_id, user_email_hash):
            logger.warning(f"Ownership check failed for session {session_id} — skipping delete.")
            return
        self._delete_session_internal(session_id)

    def enforce_session_limit(self, user_email_hash: str, max_sessions: int = MAX_SESSIONS_PER_USER):
        """
        Ensures a user has at most `max_sessions` sessions.
        Uses an atomic DELETE ... RETURNING to eliminate race conditions:
        the SELECT and DELETE execute as a single statement in one transaction,
        preventing concurrent uploads from bypassing the limit.
        """
        # Atomic delete: removes excess sessions and returns their IDs for file cleanup
        query = text(f"""
            DELETE FROM {self.full_table_name}
            WHERE session_id IN (
                SELECT session_id FROM {self.full_table_name}
                WHERE user_email_hash = :email_hash
                ORDER BY updated_at DESC
                OFFSET :max_sessions
            )
            RETURNING session_id
        """)
        with self.engine.begin() as conn:
            deleted_rows = conn.execute(query, {"email_hash": user_email_hash, "max_sessions": max_sessions}).fetchall()
        
        # Clean up associated disk files for each deleted session
        for row in deleted_rows:
            sid = row[0]
            try:
                sqlite_path = os.path.join(SQLITE_DIR, f"{sid}.db")
                if os.path.exists(sqlite_path):
                    os.remove(sqlite_path)
                    logger.info(f"Deleted SQLite file: {sqlite_path}")
                
                for f in glob.glob(os.path.join(UPLOAD_DIR, f"{sid}_*")):
                    os.remove(f)
                    logger.info(f"Deleted upload file: {f}")
                
                logger.info(f"Session limit enforced: deleted oldest session {sid} for user")
            except Exception as e:
                logger.warning(f"File cleanup failed for deleted session {sid}: {e}")

# -- helper functions --
def get_db_schema(engine: Any, table_name: str) -> str:
    """Helper to get the CREATE TABLE statement for the prompt."""
    try:
        inspector = sql_inspect(engine)
        columns = inspector.get_columns(table_name)
        
        cols_str = []
        for col in columns:
            col_type = str(col['type'])
            col_name = col['name']
            cols_str.append(f"    {col_name} {col_type}")
        
        return f"CREATE TABLE {table_name} (\n" + ",\n".join(cols_str) + "\n);"
    except Exception as e:
        logger.error(f"Failed to inspect DB schema: {e}")
        return f"Error: Could not retrieve schema for {table_name}."

def rehydrate_sqlite_from_state(state: AgentState):
    """
    Called after loading from Postgres. 
    If data exists, it rebuilds the SQLite file and Engine.
    Uses deterministic path: SQLITE_DIR/{session_id}.db
    """
    if state.get('db_engine') is not None:
        return state

    df = state.get('df_individual')
    if df is not None and not df.empty:
        
        # Try deterministic path first, then fall back to stored path
        session_id = state.get('_session_id')
        if session_id:
            db_file_path = os.path.join(SQLITE_DIR, f"{session_id}.db")
        else:
            db_file_path = state.get('db_file_path')
        
        if db_file_path and os.path.exists(db_file_path):
            logger.info(f"Reusing existing SQLite DB at: {db_file_path}")
            db_uri = f"sqlite:///{db_file_path}"
            engine = create_engine(db_uri)
            state['db_engine'] = engine
            state['db_file_path'] = db_file_path
            return state

        logger.info("Creating NEW SQLite DB from state...")

        # Determine path — use SQLITE_DIR with session_id if available
        if session_id:
            db_file_path = os.path.join(SQLITE_DIR, f"{session_id}.db")
        else:
            # Fallback for edge cases (shouldn't happen in normal flow)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False, dir=SQLITE_DIR) as tmp_db:
                db_file_path = tmp_db.name
            
        # Load Data into it
        db_uri = f"sqlite:///{db_file_path}"
        engine = create_engine(db_uri)
        
        table_name = state.get('db_table_name', 'ticket_data')
        state['df_individual'].to_sql(table_name, engine, index=False, if_exists='replace')
        
        if state.get('df_report') is not None:
             state['df_report'].to_sql("cluster_summaries", engine, index=False, if_exists='replace')

        # Update Runtime State
        state['db_file_path'] = db_file_path
        state['db_engine'] = engine
        logger.info(f"SQLite re-hydrated at: {db_file_path}")
        return state
    
    return state