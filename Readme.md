## Building RAG-based Chat APIs using Google Gemini,Pinecone, and Firebase


This project implements a Flask-based API for uploading PDF documents, indexing their content using Pinecone, and querying the indexed documents using natural language questions. The API uses Google's Generative AI for embeddings and question answering.

## Table of Contents
- [Prerequisites](#prerequisites)
- [API Endpoints](#api-endpoints)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Additional Configurations](#additional-configurations)

## Prerequisites

Before setting up the project, ensure you have the following:

- Python 3.7 or higher
- Pip (Python package manager)
- A Pinecone account and API key
- A Google Cloud account with the Generative AI API enabled
- A Firebase project with Firestore enabled

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/pdf-query-api.git
   cd pdf-query-api
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the root directory and add the following:
   ```
   FIREBASE_CREDENTIALS_PATH=/path/to/your/firebase-credentials.json
   PINECONE_API_KEY=your_pinecone_api_key
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Run the Flask application:
   ```
   python test3.py
   ```

The API should now be running on `http://localhost:5000`.

## API Endpoints

### 1. Upload Document
- **URL:** `/upload`
- **Method:** POST
- **Form Data:**
  - `file`: PDF file to upload
  - `chat-name`: Unique identifier for the chat/document
- **Response:** JSON object with upload status and details

### 2. Query Document
- **URL:** `/query`
- **Method:** POST
- **Form Data:**
  - `chat-name`: Identifier of the chat/document to query
  - `question`: The question to ask about the document
- **Response:** JSON object with the answer to the question

## Usage

1. **Uploading a Document:**
   ```
   curl -X POST -F "file=@/path/to/your/document.pdf" -F "chat-name=unique_chat_name" http://localhost:5000/upload
   ```

2. **Querying a Document:**
   ```
   curl -X POST -F "chat-name=unique_chat_name" -F "question=Your question here" http://localhost:5000/query
   ```

## Project Structure

- `test3.py`: Main Flask application file
- `requirements.txt`: List of Python dependencies
- `.env`: Environment variables (not tracked in git)
- `templates/`: Directory for HTML templates (contains `index.html`)

## Testing

To test the API:

1. Ensure the Flask application is running.
2. Use the curl commands provided in the Usage section or a tool like Postman to send requests to the API endpoints.
3. Verify that the responses are as expected.
## Additional Configurations

- **Firebase:** Ensure your Firebase credentials file is securely stored and the path is correctly set in the `.env` file.
- **Pinecone:** You may need to adjust the Pinecone index configuration in the code based on your specific needs.
- **Google Generative AI:** Make sure you have the necessary permissions and API key for using Google's Generative AI services.
