import pyttsx3 as p
import speech_recognition as sr
import google.generativeai as genai
import requests
import re

# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize pyttsx3 engine
engine = p.init()
state = True
conversation_history = []

# system_instruction = """You are Ajna, an intelligent dietary advisor designed to assist users in making informed dietary choices. When a user presents a food object to you, follow these steps: identify the food item, analyze its nutritional content, and determine its suitability for consumption based on general health guidelines. Provide detailed nutritional information, including macronutrients and micronutrients. Offer a clear and concise recommendation on whether the user should consume the food, and explain your reasoning briefly based on the food's nutritional content and potential health impacts. Ensure your responses are short and to the point, focusing on guiding the user in maintaining a healthy diet, avoiding negative health effects, and making well-informed dietary decisions."""
system_instruction = """You are Ajna, an intelligent dietary advisor designed to assist users in making informed dietary choices. When a user presents a food object to you, follow these steps: identify the food item, analyze its nutritional content, and determine its suitability for consumption based on this user medical report " **Name:** John Doe - **Age:** 45 - **Gender:** Male - **Height:** 180cm - **Weight:** 85kg - **Blood Type:** O+ - **Allergies:** Penicillin - **Chronic Conditions:** - Hypertension (diagnosed 2015) - Type 2 Diabetes (diagnosed 2018) - **Past Surgeries:** - Appendectomy (2005) - Knee arthroscopy (2017) - **Family History:** - Father: Heart disease, Type 2 Diabetes - Mother: Breast cancer - Sister: Asthma - **Current Medications:** - Metformin: 500mg twice daily - Lisinopril: 20mg daily - Aspirin: 81mg daily - **Chief Complaints:** - Shortness of breath during mild physical activities - Chest pain occasionally, especially during stress - Frequent urination and increased thirst - Fatigue and lack of energy - **Physical Examination:** - **Vital Signs:** - Blood Pressure: 150/90 mmHg - Heart Rate: 85 bpm - Respiratory Rate: 18 breaths/min - Temperature: 98.6°F - **General Appearance:** Appears overweight and slightly pale - **Cardiovascular:** Irregular heartbeat noted, no murmurs or gallops - **Respiratory:** Clear breath sounds bilaterally, no wheezing or crackles - **Abdomen:** Soft, non-tender, no organomegaly - **Extremities:** No edema, good peripheral pulses - **Laboratory Results:** - **Blood Glucose (Fasting):** 160mg/dL (Normal: 70-100mg/dL) - **HbA1c:** 7.5% (Normal: <5.7%) - **Total Cholesterol:** 240mg/dL (Normal: <200mg/dL) - **LDL Cholesterol:** 160mg/dL (Normal: <100mg/dL) - **HDL Cholesterol:** 40mg/dL (Normal: >40mg/dL) - **Triglycerides:** 180mg/dL (Normal: <150mg/dL) - **Electrocardiogram (ECG):** Irregular rhythm, signs of possible left ventricular hypertrophy - **Imaging Studies:** - **Chest X-Ray:** Mild cardiomegaly, no infiltrates - **Echocardiogram:** Ejection fraction of 45%, mild left ventricular hypertrophy - **Assessment:** 1. **Uncontrolled Hypertension:** Elevated blood pressure despite current medication. 2. **Type 2 Diabetes:** Poor glycemic control indicated by high HbA1c. 3. **Hyperlipidemia:** High total and LDL cholesterol levels. 4. **Possible Coronary Artery Disease:** Symptoms of chest pain and irregular heartbeat, suggestive of underlying cardiac issues. 5. **Obesity:** BMI indicates overweight status, contributing to metabolic syndrome. - **Plan:** 1. **Medication Adjustment:** - Increase Lisinopril to 40mg daily. - Add Atorvastatin 20mg daily for hyperlipidemia. - Add a beta-blocker for heart rate control. 2. **Lifestyle Modifications:** - Recommend a low-sodium, low-fat diet. - Increase physical activity to 30 minutes of moderate exercise 5 days a week. - Weight loss plan to reduce body weight by 10% over 6 months. 3. **Follow-Up Tests:** - Repeat blood glucose and HbA1c in 3 months. - Follow-up lipid panel in 3 months. - Stress test and possible angiography if chest pain persists. 4. **Patient Education:** - Educate about the importance of medication adherence. - Discuss the signs and symptoms of worsening heart disease. - Provide resources for smoking cessation if applicable. - **Next Appointment:** 3 months - **Physician Signature:** Dr. Jane Smith, MD". Provide detailed nutritional information, including macronutrients and micronutrients. Offer a clear and concise recommendation on whether the user should consume the food, and explain your reasoning briefly based on the food's nutritional content and potential health impacts. Ensure your responses are short and to the point, focusing on guiding the user in maintaining a healthy diet, avoiding negative health effects, and making well-informed dietary decisions."""



system_instruction2 = """You are an intelligent text processor designed to identify food items in a given sentence. Your task is to analyze the sentence, identify and extract the names of any food items or objects that are edible (such as fruits, vegetables, or any other food items), and return only the names of these food items."""

API_URL = "https://api-inference.huggingface.co/models/microsoft/git-base"
headers = {"Authorization": "Bearer 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"}

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

# Main function to start the process
def start(path):
    response = query(path)
    detected = detectobject(response,system_instruction2)
    while state==True:
        user_question = do_listen()
        if user_question == "Sorry, I didn't catch that.":
            speak("Sorry, I didn't catch that.")
            state==False
        elif user_question == "Could not request results from Google Speech Recog…":
            speak("Could not request results from Google Speech Recog…")
            state==False
        
        else:
            if user_question!=exit:
                ai_response = reason(user_question + " " + detected)
                print(ai_response)
                speak(ai_response)
            else:
                state==False

# Prompt user for object selection
object_choice = input("""Please select the fruit you choose to have:
1. Orange
2. Cabbage
3. Cauliflower
4. Watermelon
5. Banana
6. Strawberry
Enter the number corresponding to your choice: """)

# Mapping user input to file paths
file_paths = {
    "1": "./fruits/handorg.jpg",
    "2": "C:/Users/reddy/Desktop/onspot/fruits/cabbage.webp",
    "3": "C:/Users/reddy/Desktop/onspot/fruits/cauliflower.webp",
    "4": "C:/Users/reddy/Desktop/onspot/fruits/Watermelon.jpeg",
    "5": "C:/Users/reddy/Desktop/onspot/fruits/Banana.jpeg",
    "6": "C:/Users/reddy/Desktop/onspot/fruits/Strawberry.jpeg"
}

# Validate and start based on user choice
if object_choice in file_paths:
    start(file_paths[object_choice])
else:
    print("Please select an appropriate option.")
    speak("Please select an appropriate option.")
