FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install Node.js/npm so the Codex CLI can run inside the live bot container.
RUN apt-get update \
    && apt-get install -y --no-install-recommends nodejs npm \
    && npm install -g @openai/codex@0.117.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies first to optimize caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 CMD python -m ticket_execution_status || exit 1

# Run the bot script
CMD [ "python", "ysupport.py" ]
