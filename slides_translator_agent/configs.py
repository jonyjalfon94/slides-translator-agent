"""Configurations for the project."""

# %% IMPORTS

import logging
import os

# %% CONFIGS

# %% Deployment Mode
# "local" = Use Agent Engine OAuth flow (with request_credential)
# "gemini_enterprise" = Use access token from Gemini Enterprise's managed auth
DEPLOYMENT_MODE = os.getenv("DEPLOYMENT_MODE", "local")

# %% Authentication

AUTHENTICATION_CLIENT_ID = os.getenv("AUTHENTICATION_CLIENT_ID", "")
AUTHENTICATION_CLIENT_SECRET = os.getenv("AUTHENTICATION_CLIENT_SECRET", "")

# Auth ID configured in Gemini Enterprise (must match the Authorization ID in the UI)
GEMINI_ENTERPRISE_AUTH_ID = os.getenv("GEMINI_ENTERPRISE_AUTH_ID", "authentication")

# %% Concurrent

CONCURRENT_TRANSLATION_WORKERS = int(os.getenv("CONCURRENT_TRANSLATION_WORKERS", "10"))
CONCURRENT_SLIDES_BATCH_UPDATES = int(os.getenv("CONCURRENT_SLIDES_BATCH_UPDATES", "50"))

# %% Logging

LOGGING_LEVEL = getattr(logging, os.getenv("LOGGING_LEVEL", "INFO").upper(), logging.INFO)

# %% Models

MODEL_NAME_AGENT = os.getenv("MODEL_NAME_AGENT", "gemini-2.5-flash")
MODEL_NAME_TRANSLATION = os.getenv("MODEL_NAME_TRANSLATION", "gemini-2.5-flash")

# %% Projects

PROJECT_ID = os.environ["GOOGLE_CLOUD_PROJECT"]
PROJECT_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "global")

# %% Tokens

TOKEN_CACHE_KEY = os.getenv("TOKEN_CACHE_KEY", "temp:slides-translator-auth")
