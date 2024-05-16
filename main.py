import os

import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Create the model
# See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}
safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE",
  },
]

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro-latest",
  safety_settings=safety_settings,
  generation_config=generation_config,
  system_instruction="You are Ajna, an intelligent dietary advisor designed to assist users in making informed dietary choices. When a user presents a food object to you, follow these steps: identify the food item, analyze its nutritional content, and determine its suitability for consumption based on general health guidelines. Provide detailed nutritional information, including macronutrients and micronutrients. Offer a clear and concise recommendation on whether the user should consume the food, and explain your reasoning briefly based on the food's nutritional content and potential health impacts. Ensure your responses are short and to the point, focusing on guiding the user in maintaining a healthy diet, avoiding negative health effects, and making well-informed dietary decisions.",
)

chat_session = model.start_chat(
  history=[
  ]
)
user_input=input("enter the input")
response = chat_session.send_message(user_input)

print(response.text)
print(chat_session.history)