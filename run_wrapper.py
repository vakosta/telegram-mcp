import os

# Load env vars from file (written by start.sh before mcp-proxy launches)
with open('/app/.env') as f:
    for line in f:
        line = line.strip()
        if '=' in line:
            k, v = line.split('=', 1)
            os.environ[k] = v

# Run the actual MCP server
exec(open('/app/main.py').read())