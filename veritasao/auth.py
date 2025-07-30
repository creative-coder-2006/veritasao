# auth.py

"""
VERITAS.AI Authentication Module
================================

This module handles all user authentication logic, including registration,
login, and password management. It uses the bcrypt library to ensure that
passwords are securely hashed and never stored in plain text.

The Authenticator class interfaces with the SQLite database to manage user
credentials.
"""

import bcrypt
import sqlite3
from database import get_db_connection
import streamlit as st
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Authenticator:
    """
    Manages user registration, login, and session state.
    """
    def __init__(self):
        """
        Initializes the Authenticator.
        """
        logger.info("Authenticator initialized.")
        # Session state is managed by Streamlit, so no explicit state here.

    def _hash_password(self, password: str) -> bytes:
        """
        Hashes a password using bcrypt.

        Args:
            password (str): The plain-text password.

        Returns:
            bytes: The hashed password.
        """
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed

    def _verify_password(self, password: str, hashed_password: bytes) -> bool:
        """
        Verifies a plain-text password against a stored hash.

        Args:
            password (str): The plain-text password to check.
            hashed_password (bytes): The stored hashed password.

        Returns:
            bool: True if the password matches, False otherwise.
        """
        return bcrypt.checkpw(password.encode('utf-8'), hashed_password)

    def register_user(self, username: str, password: str) -> bool:
        """
        Registers a new user in the database.

        Args:
            username (str): The desired username.
            password (str): The desired password.

        Returns:
            bool: True if registration is successful, False otherwise.
        """
        if not username or not password:
            logger.warning("Registration attempt with empty username or password.")
            return False

        conn = get_db_connection()
        try:
            # Check if user already exists
            user = conn.execute('SELECT id FROM users WHERE username = ?', (username,)).fetchone()
            if user:
                logger.warning(f"Registration failed: username '{username}' already exists.")
                return False

            # Hash password and insert new user
            hashed_password = self._hash_password(password)
            conn.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            conn.commit()
            logger.info(f"Successfully registered new user: '{username}'")
            return True
        except sqlite3.Error as e:
            logger.error(f"Database error during registration for '{username}': {e}")
            return False
        finally:
            conn.close()

    def login(self, username: str, password: str) -> bool:
        """
        Logs in a user by verifying their credentials.

        Args:
            username (str): The user's username.
            password (str): The user's password.

        Returns:
            bool: True if login is successful, False otherwise.
        """
        if not username or not password:
            return False
            
        conn = get_db_connection()
        try:
            user = conn.execute('SELECT id, password FROM users WHERE username = ?', (username,)).fetchone()
            if user and self._verify_password(password, user['password']):
                # Store user info in Streamlit's session state upon successful login
                st.session_state['logged_in_user'] = username
                st.session_state['logged_in_user_id'] = user['id']
                logger.info(f"User '{username}' logged in successfully.")
                return True
            else:
                logger.warning(f"Failed login attempt for username: '{username}'")
                return False
        except sqlite3.Error as e:
            logger.error(f"Database error during login for '{username}': {e}")
            return False
        finally:
            conn.close()

    def logout(self):
        """
        Logs out the current user by clearing session state.
        """
        user = self.get_current_user()
        if 'logged_in_user' in st.session_state:
            del st.session_state['logged_in_user']
        if 'logged_in_user_id' in st.session_state:
            del st.session_state['logged_in_user_id']
        logger.info(f"User '{user}' logged out.")

    def get_current_user(self) -> str | None:
        """
        Retrieves the current logged-in user's username from session state.
        """
        return st.session_state.get('logged_in_user')

    def get_current_user_id(self) -> int | None:
        """
        Retrieves the current logged-in user's ID from session state.
        """
        return st.session_state.get('logged_in_user_id')


if __name__ == '__main__':
    # This block is for demonstration and won't run in the Streamlit app.
    # It requires a 'veritas_ai.db' to exist with the correct schema.
    print("Running Authenticator smoke test...")
    from database import init_db
    # Ensure a clean database for testing
    import os
    if os.path.exists("veritas_ai.db"):
        os.remove("veritas_ai.db")
    init_db()

    auth = Authenticator()
    test_user = "testuser"
    test_pass = "StrongPassword123"

    # Mock Streamlit session state for testing
    if 'session_state' not in globals():
        st.session_state = {}

    # --- Test Registration ---
    print(f"\nAttempting to register user '{test_user}'...")
    reg_success = auth.register_user(test_user, test_pass)
    assert reg_success, "User registration failed."
    print("Registration successful.")

    # --- Test Duplicate Registration ---
    print(f"\nAttempting to register duplicate user '{test_user}'...")
    reg_fail = not auth.register_user(test_user, test_pass)
    assert reg_fail, "Duplicate user registration should have failed."
    print("Duplicate registration correctly prevented.")

    # --- Test Login ---
    print(f"\nAttempting to log in user '{test_user}'...")
    login_success = auth.login(test_user, test_pass)
    assert login_success, "User login failed."
    assert auth.get_current_user() == test_user, "Logged in user not set in session."
    print("Login successful.")

    # --- Test Logout ---
    print("\nAttempting to log out...")
    auth.logout()
    assert auth.get_current_user() is None, "User should be logged out."
    print("Logout successful.")

    print("\nSmoke test completed successfully.")
    os.remove("veritas_ai.db") # Clean up test db