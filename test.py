from flask import Flask, request, jsonify,render_template
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv
load_dotenv()
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI    
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import os
from google.protobuf.json_format import MessageToDict  # Import protobuf to dict converter
from langchain.schema import Document
import re



FIREBASE_CREDENTIALS_PATH = os.getenv("FIREBASE_CREDENTIALS_PATH")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate(FIREBASE_CREDENTIALS_PATH)
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# @app.route('/')
# def home():
#     return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_document():
    # Check if the request contains a file and chat_name
    if 'file' not in request.files:
        return jsonify({"error": "File not provided"}), 400
    if 'chat-name' not in request.form:
        return jsonify({"error": "chat_name not provided"}), 400

    pdf_file = request.files['file']
    chat_name = request.form['chat-name']

    # Sanitize the chat_name to meet Pinecone's index naming requirements
    # sanitized_chat_name = re.sub(r'[^a-z0-9\-]', '-', chat_name.lower())

    index_name = chat_name

    # Check if the index exists, if not create it
    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=768,  # Correct dimension for the embedding model
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )

    # Extract text from the uploaded PDF using PyPDF2
    pdfreader = PdfReader(pdf_file)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content

    # Split the extracted text into manageable chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)

    # Initialize Google Generative AI Embeddings
    embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", task_type="retrieval_document")

    # Generate embeddings for each text chunk
    embeddings = embedding_model.embed_documents(texts)

    # Get Pinecone index
    index = pc.Index(index_name)
    vectors = []
    for i, (embedding, text) in enumerate(zip(embeddings, texts)):
        vectors.append({
            "id": f"{chat_name}_{i}",  # Unique ID for each vector
            "values": embedding,  # Embedding vector
            "metadata": {"text": text}  # Text chunk stored as metadata
        })

    # Upsert vectors to Pinecone
    upsert_response = index.upsert(vectors=vectors)
    upserted_count = upsert_response.upserted_count
    upsert_response_dict = MessageToDict(upsert_response)

    # Store the embeddings and text in Firestore
    doc_ref = db.collection('documents').document(chat_name)
    doc_ref.set({
        'chat_name': chat_name,
        'upserted_count': upserted_count,  # Store only relevant data
        'document_text': texts,  # Text chunks from the document
    })

    return jsonify({
        "message": "Document uploaded and indexed successfully",
        "upserted_count": upserted_count,  # Ensure upserted_count is JSON serializable
        "upsert_response": upsert_response_dict  # Return the upsert response as a Python dictionary
    }), 200

@app.route('/query', methods=['POST'])
def query_document():
    chat_name = request.form.get('chat-name')
    question = request.form.get('question')
    if validate_question(question):
        index_name = chat_name
        index = pc.Index(index_name)
        # Retrieve metadata from Firebase
        doc_ref = db.collection('documents').document(chat_name)
        doc = doc_ref.get()

        if not doc.exists:
            return jsonify({'error': 'Chat name not found'}), 404
        
        embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        embeddings = embedding_model.embed_query(question)

        
        retriever= index.query(vector=embeddings, top_k=5,include_values=False,
            include_metadata=True)
        
        documents = [Document(page_content=match['metadata']['text'], metadata=match['metadata']) for match in retriever['matches']]
        print(doc)
        return user_input(retriever=documents, question=question)
    

def get_conversational_chain():

    prompt_template = """You are a Chatbot,
    Answer the question as detailed as possible from the provided context, make sure to provide all the details,  don't provide the wrong answer\n\n
    If the question is invalid (e.g., non-sensical, offensive, or out-of-scope),return a response indicating that the question cannot be processed.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    #this prompt takes care of out-of-scope questions and offensice words
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                             temperature=0.3,api_key=GEMINI_API_KEY)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain



def user_input(retriever,question):
    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":retriever, "question": question}
        , return_only_outputs=True)

    # print(response)

    return jsonify({
        "Bot": response["output_text"]  # Return the upsert response as a Python dictionary
    }), 200

def validate_question(question):
        # Check if the input is empty or too short
    if not question or len(question.strip()) < 1:
        return False, "The question is too short. Please provide more detail."

    
    
    # Check for nonsensical input (using regex for gibberish)
    if re.match(r"^[a-zA-Z0-9]{10,}$", question):
        return False, "The question appears to be nonsensical. Please ask a meaningful question."

    
    # If all checks pass, the question is valid
    return True
    
if __name__ == '__main__':
    app.run(debug=True)
