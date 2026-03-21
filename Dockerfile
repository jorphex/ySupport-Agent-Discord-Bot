FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy and install dependencies first to optimize caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

HEALTHCHECK --interval=60s --timeout=10s --start-period=20s --retries=3 CMD python -m ticket_execution_status || exit 1

# Run the bot script
CMD [ "python", "ysupport.py" ]
