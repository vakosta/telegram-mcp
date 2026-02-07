# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Telegram MCP Server — a full-featured Telegram integration for Claude, Cursor, and any MCP-compatible client. Built with Telethon (Telegram API) and FastMCP (Model Context Protocol). Exposes 91 tools for chat management, messaging, contacts, search, folders, media, and more.

## Commands

```bash
# Install dependencies
uv sync

# Run the MCP server
uv run main.py

# Generate a Telegram session string (interactive)
uv run session_string_generator.py

# Run tests
python -m pytest test_validation.py

# Lint and format
black --check .          # line length 99, target py311
flake8 .                 # max line 99, max complexity 10

# Docker
docker compose up --build
```

## Architecture

**Single-file server** (`main.py`, ~4500 lines): FastMCP server + Telethon client + 91 `@mcp.tool()` async functions + optional webhook event system.

```
MCP Client (Claude/Cursor)
    ↓ stdio
FastMCP Server (main.py)
    ↓
TelegramClient (Telethon) → Telegram API
    ↓ (optional)
Webhook handlers → HTTP POST → WEBHOOK_URL
```

### Key patterns in main.py

- **`@validate_id()` decorator** — validates `chat_id`/`user_id` params (int, string int, or `@username`). Applied to most tools.
- **`log_and_format_error()`** — centralized error handler returning user-friendly messages with error codes (e.g. `CHAT-ERR-001`) while logging full stack traces to `mcp_errors.log` (JSON format).
- **`format_entity()` / `format_message()`** — consistent JSON serialization helpers for Telegram objects.
- **Webhook system** (lines ~345–700) — when `WEBHOOK_URL` is set, registers Telethon event handlers that POST formatted messages to an external endpoint. Currently only forwards incoming messages from other people.

### Adding a new tool

1. Add an `async def` with `@mcp.tool()` decorator (and `@validate_id()` if it takes chat/user IDs)
2. Include a docstring with `Args` — MCP uses it for parameter descriptions
3. Wrap body in `try/except` using `log_and_format_error()` for errors
4. Return a string result

## Configuration

All config via environment variables (`.env` file, loaded by `python-dotenv`):

| Variable | Required | Description |
|----------|----------|-------------|
| `TELEGRAM_API_ID` | Yes | From https://my.telegram.org/apps |
| `TELEGRAM_API_HASH` | Yes | From https://my.telegram.org/apps |
| `TELEGRAM_SESSION_STRING` | One of | String session (preferred, more portable) |
| `TELEGRAM_SESSION_NAME` | these | File-based session name |
| `WEBHOOK_URL` | No | POST endpoint for Telegram event notifications |
| `WEBHOOK_API_KEY` | No | Bearer token for webhook auth |

## Code Style

- **Black** with line length 99, target Python 3.11
- **Flake8** with max line 99, max complexity 10
- Logger level is `ERROR` in production (lines 84–88 in main.py)
- Prefer string sessions over file-based sessions (avoids SQLite lock issues)
