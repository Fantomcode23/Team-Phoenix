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
import os
import time
import random
import pyttsx3 as p
import speech_recognition as sr
import google.generativeai as genai
import requests
import re
import keyboard
import threading
import time
API_URL = "https://api-inference.huggingface.co/models/microsoft/git-base"
headers = {"Authorization": "Bearer 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"}
# -----------------------------------------------firebase  download -----------------------------------------
import firebase_admin
from firebase_admin import credentials,storage

recognizer = sr.Recognizer()

# Initialize pyttsx3 engine
engine = p.init()
state = True
conversation_history = []

global pdf_name

# Initialize Firebase app with service account credentials
cred = credentials.Certificate("C:\\Users\\reddy\\Desktop\\ajnaweb\\team-phoenix-49d24-1293b08fc298.json")
firebase_admin.initialize_app(cred)

bucket_name = "fcode-46962.appspot.com"

def download(cloud_file_name,local_file_path):
    bucket = storage.bucket(bucket_name)
    cloud_file_name = cloud_file_name
    local_file_path = local_file_path

    try:
        blob = bucket.blob(cloud_file_name)
        blob.download_to_filename(local_file_path)
        print(f"File '{cloud_file_name}' fetched and saved to '{local_file_path}' successfully!")
    except Exception as e:
        print("Error fetching file:", e)

# ------------------------------------------------------------------------------------------------
destination_blob_name = 'storage'

# --------------------------------------------------upload---------------------------------------------

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

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    print(text)
    return text


def upload_file(file_path, destination_blob_name):
    """
    Uploads a file to the bucket.
    :param file_path: Path to the file to upload.
    :param destination_blob_name: Name of the destination blob in the bucket.
    """
    bucket = storage.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_path)
    print(f"File {file_path} uploaded to {destination_blob_name}.")



@app.route('/upload', methods=['POST'])
def upload():
    global pdf_name
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        pdf_name = os.path.basename(file_path)
        upload_file(upload_file,destination_blob_name)
        print("file uploaded to firebase")
        file.save(file_path)
        session['uploaded_file_path'] = file_path
        return redirect(url_for('upload'))

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    global report 
    prompt_template = """
    you are a proffessional report analyser to give the exact contnet of the report in each and every part of the medical report which is feed as context 
    Context:\n {context}?\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    print("i am in get_conversational_chain")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    report=chain
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    print("\n \n i am in chunk function \n \n")
    print(chunks)
    return chunks

@app.route('/proceed')
def proceed():
    if 'uploaded_file_path' not in session:
        return 'No file uploaded'
    # uploaded_file_path = session['uploaded_file_path']
    uploaded_file_path=download(pdf_name,)
    raw_text = get_pdf_text(uploaded_file_path)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(text_chunks)
    return redirect(url_for('conversate'))

@app.route('/conversate', methods=['GET', 'POST'])
def conversate():
    global report
    response = ""
    if request.method == 'POST':
        user_question = request.form['input']
        report=user_input(user_question="i want each and every pointof thep point in this context")
        response=reason(user_question)

    return render_template('conversate.html', response=response)

if __name__ == '__main__':
    app.run(debug=True)


# =----------------------------------------------


#  ----------------------------------------------------------------------------------------------------

# Initialize speech recognizer



system_instruction = f"""You are Ajna, an intelligent dietary advisor designed to assist users in making informed dietary choices. When a user presents a food object to you, follow these steps: identify the food item, analyze its nutritional content, and determine its suitability for consumption based on this user medical report"{report}". Provide detailed nutritional information, including macronutrients and micronutrients. Offer a clear and concise recommendation on whether the user should consume the food, and explain your reasoning briefly based on the food's nutritional content and potential health impacts. Ensure your responses are short and to the point, focusing on guiding the user in maintaining a healthy diet, avoiding negative health effects, and making well-informed dietary decisions."""



system_instruction2 = """You are an intelligent text processor designed to identify food items in a given sentence. Your task is to analyze the sentence, identify and extract the names of any food items or objects that are edible (such as fruits, vegetables, or any other food items), and return only the names of these food items."""



def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    response_json = response.json()
    print(response_json[0]["generated_text"])
    return (response_json[0]["generated_text"])


def clean_text(text):
    # Remove ** and any other unwanted symbols
    cleaned_text = re.sub(r"[*_~`]", "", text)
    return cleaned_text


def speak(text):
    cleaned_text = clean_text(text)
    engine.say(cleaned_text)
    engine.runAndWait()


def do_listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise and energy threshold
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 400  # Adjust this threshold as needed
        print("Listening...")
        audio = recognizer.listen(source)
        
        try:
            text = recognizer.recognize_google(audio)
            print("User:", text)
            return text.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
            return "Sorry, I didn't catch that."
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))
            return "Could not request results from Google Speech Recognition service;"

# Configure the AI assistant
genai.configure(api_key="AIzaSyANF58fvFynlOZM1DzXpWoUmin6UV99mcI")

# Set up the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Object detection function
def detectobject(input_text, system_instruction2):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  system_instruction=system_instruction2,
                                  safety_settings=safety_settings)
    convo = model.start_chat(history=[
    ])
    convo.send_message(input_text)
    detected = convo.last.text
    return detected

# Response generation function
def reason(user_question):
    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                  generation_config=generation_config,
                                  system_instruction=system_instruction,
                                  safety_settings=safety_settings)
    convo = model.start_chat(history=conversation_history)
    convo.send_message(user_question)
    model_response = convo.last.text
    conversation_history.append({"role": "model", "parts": [model_response]})
    return model_response





