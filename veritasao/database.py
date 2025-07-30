# database.py

"""
VERITAS.AI Database Module
==========================

This module handles all interactions with the SQLite database. It is responsible
for:
-   Initializing the database and creating the necessary tables.
-   Storing user credentials (managed by the auth module).
-   Logging all analysis reports for user history.
-   Tracking user-generated flags for unreliable sources.
"""

import sqlite3
from config import DATABASE_CONFIG
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_FILE = DATABASE_CONFIG.get("db_file", "veritas_ai.db")

def get_db_connection():
    """
    Establishes a connection to the SQLite database.
    Configured to return rows as dictionary-like objects.

    Returns:
        sqlite3.Connection: A database connection object.
    """
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database by creating tables if they don't already exist.
    This function is idempotent and safe to run on every application startup.
    """
    conn = get_db_connection()
    try:
        with conn:
            # --- Users Table ---
            # Stores user credentials for login.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            ''')

            # --- Analysis History Table ---
            # Logs every analysis performed by a user.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    url TEXT NOT NULL,
                    platform TEXT NOT NULL,
                    scores TEXT, -- Storing the scores dictionary as a JSON string
                    explanation TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                );
            ''')

            # --- Source Flags Table ---
            # Tracks sources that users have flagged as unreliable.
            conn.execute('''
                CREATE TABLE IF NOT EXISTS source_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    flagged_by_user_id INTEGER NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(domain, flagged_by_user_id), -- A user can only flag a domain once
                    FOREIGN KEY (flagged_by_user_id) REFERENCES users (id)
                );
            ''')
        logger.info("Database initialized successfully. Tables created or verified.")
    except sqlite3.Error as e:
        logger.error(f"An error occurred during database initialization: {e}")
    finally:
        conn.close()

def add_analysis_record(user_id: int, url: str, platform: str, scores: str, explanation: str):
    """
    Adds a new analysis record to the history table.

    Args:
        user_id (int): The ID of the user who performed the analysis.
        url (str): The URL of the content that was analyzed.
        platform (str): The platform of the content (e.g., 'News', 'YouTube').
        scores (str): A JSON string representing the scores dictionary.
        explanation (str): The AI-generated explanation text.
    """
    conn = get_db_connection()
    try:
        with conn:
            conn.execute('''
                INSERT INTO analysis_history (user_id, url, platform, scores, explanation, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (user_id, url, platform, scores, explanation, datetime.now()))
        logger.info(f"Added analysis record for user {user_id} and URL {url}")
    except sqlite3.Error as e:
        logger.error(f"Failed to add analysis record: {e}")
    finally:
        conn.close()

def get_user_history(user_id: int) -> list:
    """
    Retrieves the analysis history for a specific user.

    Args:
        user_id (int): The ID of the user.

    Returns:
        list: A list of dictionary-like rows representing the user's history, ordered by most recent.
    """
    conn = get_db_connection()
    try:
        history = conn.execute('''
            SELECT url, platform, scores, strftime('%Y-%m-%d %H:%M:%S', timestamp) as timestamp
            FROM analysis_history
            WHERE user_id = ?
            ORDER BY timestamp DESC
        ''', (user_id,)).fetchall()
        return [dict(row) for row in history]
    except sqlite3.Error as e:
        logger.error(f"Failed to retrieve history for user {user_id}: {e}")
        return []
    finally:
        conn.close()

def flag_source(domain: str, username: str) -> bool:
    """
    Flags a domain as unreliable for a specific user.

    Args:
        domain (str): The domain to be flagged (e.g., 'example.com').
        username (str): The username of the user flagging the source.

    Returns:
        bool: True if the flag was added successfully, False if it was a duplicate.
    """
    conn = get_db_connection()
    try:
        with conn:
            # First, get the user ID from the username
            user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            if not user:
                logger.warning(f"Attempt to flag source by non-existent user '{username}'")
                return False
            user_id = user['id']

            # Insert the flag, ignoring if it's a duplicate (due to UNIQUE constraint)
            conn.execute('''
                INSERT OR IGNORE INTO source_flags (domain, flagged_by_user_id, timestamp)
                VALUES (?, ?, ?)
            ''', (domain, user_id, datetime.now()))
            # The `changes()` function returns the number of rows modified by the last statement.
            # If it's 0, it means the INSERT was ignored (i.e., duplicate).
            if conn.total_changes == 0:
                logger.warning(f"User '{username}' already flagged domain '{domain}'.")
                return False
        logger.info(f"Domain '{domain}' flagged by user '{username}'.")
        return True
    except sqlite3.Error as e:
        logger.error(f"Failed to flag source '{domain}': {e}")
        return False
    finally:
        conn.close()

def get_source_flags(domain: str) -> int:
    """
    Counts how many times a domain has been flagged by all users.

    Args:
        domain (str): The domain to check.

    Returns:
        int: The total number of flags for the given domain.
    """
    conn = get_db_connection()
    try:
        count = conn.execute('SELECT COUNT(id) FROM source_flags WHERE domain = ?', (domain,)).fetchone()[0]
        return count
    except sqlite3.Error as e:
        logger.error(f"Failed to get flag count for domain '{domain}': {e}")
        return 0
    finally:
        conn.close()

if __name__ == '__main__':
    # This block is for testing and demonstration.
    print("Running database module smoke test...")
    # Create a fresh in-memory database for testing
    DB_FILE = ":memory:"
    init_db()
    print("In-memory database initialized.")

    # You would typically call auth functions to create a user first.
    # For this test, we'll manually insert one.
    conn = get_db_connection()
    conn.execute("INSERT INTO users (username, password) VALUES ('testuser', 'hashed_pass')")
    user_id = conn.execute("SELECT id FROM users WHERE username = 'testuser'").fetchone()['id']
    conn.close()
    print(f"Manually inserted 'testuser' with ID: {user_id}")
    
    # --- Test adding and getting history ---
    add_analysis_record(user_id, "http://googleusercontent.com/1", "News", '{"trust_score": 0.5}', "Explanation text.")
    history = get_user_history(user_id)
    assert len(history) == 1
    assert history[0]['platform'] == "News"
    print("Analysis record added and retrieved successfully.")

    # --- Test flagging ---
    flag_success = flag_source("unreliable-site.com", "testuser")
    assert flag_success
    flag_duplicate = not flag_source("unreliable-site.com", "testuser")
    assert flag_duplicate
    flag_count = get_source_flags("unreliable-site.com")
    assert flag_count == 1
    print("Source flagging system works as expected.")

    print("\nSmoke test completed successfully.")