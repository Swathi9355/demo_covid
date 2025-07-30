import google.generativeai as genai

genai.configure(api_key="AIzaSyAwar6hRUoyt_Co9ONNp1hvWjQXP3w1KBs")

model=genai.GenerativeModel(model_name="gemini-2.0-flash")

chat=model.start_chat(history=[])
while True:
    prmt=input("hello welcome to my channel")
    if(prmt=="exit"):
        break


res=chat.send_message(prmt)
print(res.text)