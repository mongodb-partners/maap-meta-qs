# AI Memory Documentation

## Table of Contents
- [1. Overview](#1-overview)
- [2. System Architecture](#2-system-architecture)
- [3. Components](#3-components)
- [4. Installation & Deployment](#4-installation--deployment)
- [5. Configuration](#5-configuration)
- [6. Usage](#6-usage)
- [7. API Reference](#7-api-reference)
- [8. Security Considerations](#8-security-considerations)
- [9. Monitoring & Logging](#9-monitoring--logging)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Development Guide](#11-development-guide)
- [12. Maintenance & Operations](#12-maintenance--operations)

## 1. Overview

AI Memory is a comprehensive solution for managing conversation history and providing contextual memory for AI applications. The system enables storing, retrieving, and searching through conversation data using both semantic (vector) and full-text search capabilities. It leverages advanced embedding models to create rich representations of text, allowing for intelligent retrieval of relevant conversation snippets when needed.

Key features include:
- Conversation memory storage and retrieval
- Hybrid search combining vector and full-text search capabilities
- Context-aware memory retrieval
- Conversation summarization
- Support for both synchronous and asynchronous operations

The primary use case is enhancing AI assistants with long-term memory, allowing them to recall previous interactions and provide more consistent and contextually relevant responses.

## 2. System Architecture

```mermaid
graph TD
    %% Main Classes
    A[AIMemory Class] --> A1[Singleton Pattern]
    A --> A2[HTTP Client Management]
    A --> A3[Async/Sync API Methods]
    
    %% Server Components
    B[FastAPI Server] --> B1[MongoDB Integration]
    B --> B2[AWS Bedrock Integration]
    B --> B3[Vector & Text Search]
    
    %% Memory Processing Flow
    C[Message Processing] --> C1[Embedding Generation]
    C --> C2[(MongoDB Storage)]
    
    %% API Endpoints
    D[API Endpoints] --> D1[conversation]
    D --> D2[search]
    D --> D3[retrieve_memory]
    D --> D4[generate_conversation_summary]
    D --> D5[get_conversation_context]
    
    %% Client Usage
    E[Client Usage] --> E1[Async Mode]
    E --> E2[Sync Mode]
    
    %% Client-Server Flow
    A3 -->|HTTP Requests| B
    B -->|Responses| A3
    
    %% Message Flow
    E1 -->|Messages| A3
    E2 -->|Messages| A3
    A3 -->|Store| D1
    D1 -->|Process| C
    C1 -->|Vector| C2
    
    %% Search & Retrieval Flow
    D2 -->|Query| C1
    D2 -->|Search| B3
    D3 -->|Semantic Search| B3
    D3 -->|Get Context| D5
    D3 -->|Generate Summary| D4
    D4 -->|LLM Request| B2
    
    %% Hybrid Search Process
    B3 -->|Vector Search| F[Hybrid Search]
    B3 -->|Text Search| F
    F -->|Results| D2
    F -->|Results| D3
    
    %% MongoDB Operations
    B1 -->|Store Messages| C2
    B1 -->|Create Indexes| G[MongoDB Indexes]
    G -->|Vector Index| B3
    G -->|Text Index| B3
    
    %% Legend/Notes
    subgraph "Client Library (memory.py)"
        A
        A1
        A2
        A3
    end
    
    subgraph "Backend Server (main.py)"
        B
        B1
        B2
        B3
        C
        C1
        C2
        D
        D1
        D2
        D3
        D4
        D5
        F
        G
    end
    
    subgraph "Client Implementation (example.py)"
        E
        E1
        E2
    end
```


The architecture consists of several key components:
1. **FastAPI Server**: Serves as the main API endpoint for all memory operations
2. **MongoDB Database**: Stores conversation data with vector embeddings
3. **AWS Bedrock**: Provides embedding generation and LLM capabilities
4. **AIMemory Client**: A lightweight client for applications to interact with the memory service

## 3. Components

### 3.1 FastAPI Server (main.py)

**Purpose**: Provides a RESTful API interface for all memory operations.

**Technologies**:
- FastAPI: Web framework for building APIs
- Uvicorn: ASGI server for running FastAPI
- PyMongo: MongoDB client for Python
- Boto3: AWS SDK for Python

**Core Functionality**:
- Conversation storage and retrieval
- Memory search using hybrid search techniques
- Context retrieval around specific conversation points
- Conversation summarization

**Interactions**:
- Receives HTTP requests from client applications
- Connects to MongoDB for data storage and retrieval
- Communicates with AWS Bedrock for AI capabilities

### 3.2 AIMemory Client (memory.py)

**Purpose**: Provides a simple client interface for applications to interact with the memory service.

**Technologies**:
- HTTPX: Asynchronous HTTP client
- AsyncIO: Asynchronous programming support

**Core Functionality**:
- Synchronous and asynchronous HTTP requests
- Singleton pattern implementation for consistent client state
- Error handling and request management

**Interactions**:
- Connects to the FastAPI server
- Provides both synchronous and asynchronous interfaces for applications

### 3.3 MongoDB Database

**Purpose**: Stores conversation data and provides search capabilities.

**Key Features**:
- Vector search for semantic similarity
- Full-text search for keyword matching
- Document storage for conversation history

**Collections**:
- `conversations`: Stores message content, metadata, and vector embeddings

## 4. Installation & Deployment

### Prerequisites
- Python 3.8+
- MongoDB Atlas account or local MongoDB instance (with vector search capability)
- AWS account with Bedrock access

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/mohammaddaoudfarooqi/ai-memory
cd ai-memory
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Deployment Steps

1. Set up MongoDB:
   - Create a MongoDB Atlas cluster or set up a local MongoDB instance
   - Enable vector search capabilities

2. Configure AWS Bedrock:
   - Set up an AWS account with Bedrock access
   - Create access keys for API access

3. Configure environment variables:
   - Copy `sample.env` to `.env`
   - Fill in the required credentials (see Configuration section)

4. Start the server:
```bash
python main.py
```

The server will start on `http://0.0.0.0:8182` by default.

## 5. Configuration

The application uses environment variables for configuration, which can be set in a `.env` file:

```
MONGODB_URI="mongodb+srv://username:password@cluster.mongodb.net/?retryWrites=true&w=majority"
AWS_ACCESS_KEY_ID="your_aws_access_key"
AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
AWS_REGION="us-east-1"
LLM_MODEL_ID="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
```

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| MONGODB_URI | Connection string for MongoDB | Yes | None |
| AWS_ACCESS_KEY_ID | AWS access key for Bedrock | Yes | None |
| AWS_SECRET_ACCESS_KEY | AWS secret key for Bedrock | Yes | None |
| AWS_REGION | AWS region for Bedrock | Yes | "us-east-1" |
| LLM_MODEL_ID | Model ID for AWS Bedrock | Yes | "us.anthropic.claude-3-7-sonnet-20250219-v1:0" |

## 6. Usage

### Basic Usage with AIMemory Client

```python
from memory import AIMemory

# Create an instance (singleton pattern ensures only one instance)
ai_memory = AIMemory(base_url="http://localhost:8182")

# Define user and conversation IDs
USER_ID = "user@example.com"
CONVERSATION_ID = "conversation_1"

# Store a human message
ai_memory.post(
    f"/conversation/{USER_ID}/{CONVERSATION_ID}", 
    data={"type": "human", "text": "Hello, can you help me with Python?"}
)

# Store an AI response
ai_memory.post(
    f"/conversation/{USER_ID}/{CONVERSATION_ID}", 
    data={"type": "ai", "text": "Sure! What would you like to know about Python?"}
)

# Retrieve memory based on a query
response = ai_memory.get(
    "/retrieve_memory/", 
    {"user_id": USER_ID, "text": "Python help"}
)
```

### Asynchronous Usage

```python
import asyncio
from memory import AIMemory

ai_memory = AIMemory(base_url="http://localhost:8182")

async def main():
    # Store a message asynchronously
    await ai_memory.apost(
        f"/conversation/{USER_ID}/{CONVERSATION_ID}", 
        data={"type": "human", "text": "Hello"}
    )
    
    # Retrieve memory asynchronously
    response = await ai_memory.aget(
        "/retrieve_memory/", 
        {"user_id": USER_ID, "text": "Hello"}
    )
    
    # Close the client connection
    await ai_memory.close()

asyncio.run(main())
```

## 7. API Reference

### Conversation Endpoints

#### Add Message to Conversation
```
POST /conversation/{user_id}/{conversation_id}
```
**Parameters:**
- `user_id`: String - Unique identifier for the user
- `conversation_id`: String - Identifier for the conversation

**Request Body:**
```json
{
  "type": "human|ai",
  "text": "Message content",
  "timestamp": "2023-05-01T12:00:00Z" (optional)
}
```

**Response:**
```json
{
  "message": "Message added successfully."
}
```

#### Retrieve Conversation
```
GET /conversation/
```
**Query Parameters:**
- `user_id`: String - Unique identifier for the user
- `conversation_id`: String - Identifier for the conversation

**Response:**
```json
{
  "documents": [
    {
      "type": "human|ai",
      "text": "Message content",
      "timestamp": "2023-05-01T12:00:00Z"
    }
  ]
}
```

### Memory Retrieval Endpoints

#### Search Memory
```
GET /search/
```
**Query Parameters:**
- `user_id`: String - Unique identifier for the user
- `query`: String - Text to search for

**Response:**
```json
{
  "documents": [
    {
      "_id": "document_id",
      "fts_score": 0.95,
      "vs_score": 0.87,
      "score": 0.91,
      "text": "Message content",
      "type": "human|ai",
      "timestamp": "2023-05-01T12:00:00Z",
      "conversation_id": "conversation_1"
    }
  ]
}
```

#### Retrieve Memory with Context
```
GET /retrieve_memory/
```
**Query Parameters:**
- `user_id`: String - Unique identifier for the user
- `text`: String - Text to search for

**Response:**
```json
{
  "memory_items": {
    "documents": [...]
  },
  "context": {
    "documents": [...]
  },
  "summary": {
    "summary": "Conversation summary text"
  }
}
```

#### Get Conversation Context
```
GET /get_conversation_context/
```
**Query Parameters:**
- `_id`: String - Document ID to get context around

**Response:**
```json
{
  "documents": [
    {
      "type": "human|ai",
      "text": "Message content",
      "timestamp": "2023-05-01T12:00:00Z",
      "user_id": "user@example.com",
      "conversation_id": "conversation_1"
    }
  ]
}
```

#### Generate Conversation Summary
```
GET /generate_conversation_summary/
```
**Query Parameters:**
- `documents`: String - JSON string of conversation documents

**Response:**
```json
{
  "summary": "Generated summary of the conversation"
}
```

## 8. Security Considerations

### Authentication and Authorization
- The current implementation does not include built-in authentication
- It is recommended to implement an authentication layer (e.g., API keys, OAuth) when deploying to production
- Consider using API gateways or reverse proxies with authentication capabilities

### Data Protection
- Sensitive conversation data is stored in MongoDB
- Ensure MongoDB connection uses TLS/SSL encryption
- Use MongoDB Atlas security features like IP whitelisting and VPC peering
- Consider encrypting sensitive data fields

### API Security
- Deploy behind a secure gateway or load balancer
- Implement rate limiting to prevent abuse
- Use HTTPS for all communications

### AWS Credentials
- Store AWS credentials securely using environment variables
- Use IAM roles with minimal required permissions
- Regularly rotate access keys

## 9. Monitoring & Logging

### Logging
The application uses Python's built-in logging module with the following configuration:
- Log level: INFO
- Log format includes timestamps and log levels

Example log output:
```
INFO:     Started server process [12345]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8182
```

### Monitoring Considerations
- Monitor MongoDB performance using MongoDB Atlas metrics
- Track AWS Bedrock usage and quotas
- Consider implementing application metrics using Prometheus or similar tools
- Set up alerts for API errors and latency issues

### Key Metrics to Track
- API endpoint response times
- Memory search latency
- MongoDB query performance
- AWS Bedrock API call success rates
- Error rates by endpoint

## 10. Troubleshooting

### Common Issues and Solutions

#### Connection Issues with MongoDB
**Symptoms**: Server fails to start, MongoDB connection errors
**Solutions**:
- Verify MongoDB URI is correct in the `.env` file
- Check network connectivity to MongoDB server
- Ensure IP address is whitelisted in MongoDB Atlas

#### AWS Bedrock Access Issues
**Symptoms**: Embedding generation fails, model inference fails
**Solutions**:
- Verify AWS credentials are correct
- Check if the specified model is available in your AWS region
- Ensure your AWS account has access to the Bedrock service and models

#### API Endpoint Failures
**Symptoms**: HTTP 500 errors when calling endpoints
**Solutions**:
- Check server logs for detailed error messages
- Verify request format matches API specifications
- Ensure MongoDB indexes are properly created

#### Memory Search Not Returning Expected Results
**Symptoms**: Search queries don't return relevant results
**Solutions**:
- Adjust the weight parameter in hybrid search (default is 0.5)
- Check if embeddings are being properly generated
- Ensure text is properly normalized before search

### Debugging
- Enable debug logging by setting the log level to DEBUG
- Use FastAPI's built-in `/docs` endpoint to test API calls directly
- Check MongoDB logs for query performance issues

## 11. Development Guide

### Codebase Organization

```
ai-memory/
├── main.py           # FastAPI server implementation
├── memory.py         # AIMemory client library
├── example.py        # Example usage of the client
└── sample.env        # Environment variable template
```

### Key Files and Their Purpose

- **main.py**: Core server implementation with API endpoints and MongoDB integration
- **memory.py**: Client library for interacting with the memory service
- **example.py**: Example usage demonstrating both synchronous and asynchronous patterns
- **sample.env**: Template for environment variables

### Development Environment Setup

1. Set up a development environment following the installation instructions
2. For local testing, consider using MongoDB Community Edition
3. Use FastAPI's automatic documentation at `/docs` for API testing

### Testing

For manual testing:
1. Start the server with `python main.py`
2. Use the FastAPI Swagger UI at `http://localhost:8182/docs`
3. Execute the example script with `python example.py`

For automated testing, consider implementing:
- Unit tests for individual functions
- Integration tests for API endpoints
- Mock objects for MongoDB and AWS Bedrock

## 12. Maintenance & Operations

### Regular Maintenance Tasks

1. **Database Maintenance**:
   - Monitor MongoDB disk usage and performance
   - Optimize indexes periodically
   - Check for slow queries and optimize as needed

2. **Dependency Updates**:
   - Regularly update Python packages
   - Monitor for security vulnerabilities in dependencies
   - Test updates in a staging environment before production deployment

3. **Backup Procedures**:
   - Set up regular MongoDB backups
   - Store backups securely and test restoration procedures

### Scaling Considerations

1. **Horizontal Scaling**:
   - Deploy multiple instances of the FastAPI server behind a load balancer
   - Use MongoDB Atlas scaling capabilities for database growth

2. **Performance Optimization**:
   - Use connection pooling for MongoDB
   - Implement caching for frequently accessed data
   - Optimize embedding generation and search algorithms

### Monitoring Long-term Health

1. **System Health Checks**:
   - Implement endpoint health checks (`/health`)
   - Monitor system resource usage (CPU, memory, disk)
   - Set up alerts for abnormal patterns

2. **Usage Analytics**:
   - Track API usage patterns
   - Monitor storage growth over time
   - Analyze search query performance and relevance

3. **Documentation Updates**:
   - Keep documentation in sync with code changes
   - Document known issues and workarounds
   - Maintain a changelog for major updates