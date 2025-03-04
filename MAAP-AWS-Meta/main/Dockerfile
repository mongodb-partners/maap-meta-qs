FROM python:3.10-slim

COPY ./requirements.txt /code/requirements.txt
COPY ./main.py /code/main.py
COPY ./agents.py /code/agents.py
COPY ./memory.py /code/memory.py
COPY ./orchestrator.py /code/orchestrator.py
COPY ./logger.py /code/logger.py
COPY ./semantic_cache.py /code/semantic_cache.py
COPY ./cache.py /code/cache.py
COPY ./tools.py /code/tools.py
COPY .env /code/.env

# Set up working directory
WORKDIR /code

# Install dependencies as root
RUN pip install -r requirements.txt

# Expose the FastAPI default port
EXPOSE 8000

# Run the application
ENTRYPOINT [ "python3", "main.py" ]