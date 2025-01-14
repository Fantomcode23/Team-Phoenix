import pyttsx3 as p
import speech_recognition as sr


# Initialize speech recognizer
recognizer = sr.Recognizer()

# Initialize pyttsx3 engine
engine = p.init()



def speak(text):
    engine.say(text)
    engine.runAndWait()


def do_listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust for ambient noise and energy threshold
        recognizer.adjust_for_ambient_noise(source, duration=1)
        recognizer.energy_threshold = 300  # Adjust this threshold as needed
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


while(1):
    test_user=do_listen()
    if test_user:
        speak(test_user)
    

    



