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
import cv2

import pyttsx3 as p
import speech_recognition as sr
import google.generativeai as genai
import requests
import re
import keyboard
import threading
import time
# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize pyttsx3 engine
engine = p.init()
state = True
conversation_history = []
# Initialize Firebase app with service account credentials
cred = credentials.Certificate("C:\\Users\\reddy\\Desktop\\ajnaweb\\team-phoenix-49d24-1293b08fc298.json")
firebase_admin.initialize_app(cred)

bucket_name = "team-phoenix-49d24.appspot.com"

app = Flask(_name_)
app.secret_key = 'ajnakey'  # Necessary for using session

load_dotenv()
genai.configure(api_key="AIzaSyANF58fvFynlOZM1DzXpWoUmin6UV99mcI")
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
    if request.method == 'POST':
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
            return redirect(url_for('home'))
    return redirect(url_for('home'))

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
    you are a proffessional report analyser to give the exact contnet of the report in each and every part of the medical report which is feed as context 
    Context:\n {context}?\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.9)
    print("i am in get_conversational_chain")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
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
    return redirect(url_for('proceed'))

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


@app.route('/conversate', methods=['GET', 'POST'])
def conversate():
    global report
    global system_instruction
    response = ""
    if request.method == 'POST':
        user_question = request.form['input']

        report = user_input(user_question)
        system_instruction = f"""You are Ajna, an intelligent dietary advisor designed to assist users in making informed dietary choices. When a user presents a food object to you, follow these steps: identify the food item, analyze its nutritional content, and determine its suitability for consumption based on this user medical report {report}. Provide detailed nutritional information, including macronutrients and micronutrients. Offer a clear and concise recommendation on whether the user should consume the food, and explain your reasoning briefly based on the food's nutritional content and potential health impacts. Ensure your responses are short and to the point, focusing on guiding the user in maintaining a healthy diet, avoiding negative health effects, and making well-informed dietary decisions. what ever  question the user  ask you  analyse the report of the user and answer based on the medical report given  and always be consise and never give lengthy replies."""
        response=reason(user_question)
    return render_template('index.html',response=response)

# --------------------------------------------------------------camera part ---------------------------------
def capture_image(camera_index=0, image_path="captured_image.jpg"):
    # Open the camera
    cap = cv2.VideoCapture(camera_index)

    # Check if the camera is opened successfully
    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Create a window to display the video feed
    cv2.namedWindow("Video Feed", cv2.WINDOW_NORMAL)

    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        # Check if the frame is read successfully
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Display the frame
        cv2.imshow("Video Feed", frame)

        # Check for key press
        key = cv2.waitKey(1)

        # Capture image when space key is pressed
        if key == 32:  # 32 is the ASCII code for space key
            # Save the captured image
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved to {image_path}")

        # Break the loop if the 'q' key is pressed
        if key == ord('q'):
            break

    # Release the camera and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


@app.route('/capture_image', methods=['POST'])
def capture_image_route():
    capture_image()
    return redirect(url_for('home'))




if _name_ == '_main_':
    app.run(debug=True)

