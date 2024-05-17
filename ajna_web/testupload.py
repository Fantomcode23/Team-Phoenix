from flask import Flask, render_template, request, redirect, url_for, session
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import credentials, storage

# Initialize Firebase app with service account credentials
cred = credentials.Certificate("C:\\Users\\reddy\\Desktop\\ajnaweb\\team-phoenix-49d24-1293b08fc298.json")
firebase_admin.initialize_app(cred)

bucket_name = "team-phoenix-49d24.appspot.com"

app = Flask(__name__)
app.secret_key = 'ajnakey'  # Necessary for using session

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the uploads directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    return render_template('index.html')

def upload_file(file_path, destination_blob_name):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")

def download_file(cloud_file_name, local_file_path):
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(cloud_file_name)
    blob.download_to_filename(local_file_path)
    print(f"File '{cloud_file_name}' fetched and saved to '{local_file_path}'.")

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        upload_file(file_path, file.filename)  # Upload the file to Firebase Storage
        session['uploaded_file_name'] = file.filename
        return redirect(url_for('proceed'))

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print("\n \n i am in chunk function \n \n")
    print(chunks)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, and based on the condition of the user medical report mentioned you need to have what kind of food products will affect the user,if the user is asking irrelevent question other than the health or normal conversation then you say that "you are not expert in other things so please so ask if you have any health query or any thing related to health you have to answer the question asked and give him the advice whether he can eat the food item or not\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    print("i am in get_conversational_chain")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

@app.route('/proceed')
def proceed():
    if 'uploaded_file_name' not in session:
        return 'No file uploaded'
    uploaded_file_name = session['uploaded_file_name']
    local_file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file_name)
    download_file(uploaded_file_name, local_file_path)  # Fetch the file from Firebase Storage
    raw_text = get_pdf_text(local_file_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return redirect(url_for('conversate'))

@app.route('/conversate', methods=['GET', 'POST'])
def conversate():
    response = ""
    if request.method == 'POST':
        user_question = request.form['input']
        response = user_input(user_question)
    return render_template('conversate.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)
