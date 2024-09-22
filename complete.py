import cv2
import os
import pytesseract
import numpy as np
import google.generativeai as genai
import streamlit as st
import time
from PIL import Image
from otherImgGen import ImageGenerator as IG


# Configure API key and initialize model
genai.configure(api_key="AIzaSyCD6M571IvBJHm31wTF5vOrGV60gk-PtRQ")
model = genai.GenerativeModel('gemini-1.5-flash')

def stream_data(data, delay: float = 0.1):
    placeholder = st.empty()  # Create an empty placeholder to update text
    text = ""
    for word in data.split():
        text += word + " "
        placeholder.markdown(f"""<p style="color:rgb(27, 233, 212)">{text}</p>""", unsafe_allow_html= True)  # Display progressively in markdown
        time.sleep(delay)


def recognize_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='hin+eng')            # lang='hin+eng'
        return text.strip()
    except Exception as e:
        print(e)
        return None

def summarize_text(text):
    try:
        response = model.generate_content(f"Summarize the following text: {text}")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(e)

def prompt(text):
    try:
        response = model.generate_content(f"Generate a prompt for image generation for the given text: {text}. Only generate the single best prompt and nothing else.")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(e)

def translate(summary, lang):
    try:
        response = model.generate_content(f"Translate the following {summary} into {lang} and use simple words.")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        print(e)

def processing(file):
    prompt_list = []
    
    # Convert PIL image to NumPy array for OpenCV processing
    image = np.array(file)
    
    # Recognize text from the image
    texts = recognize_text(image)
    print(f'Recognized texts: {texts}')
    
    # Summarize the recognized text
    summary = summarize_text(texts)
    print(f'Summary: {summary}')
    stream_data(summary, 0.02)
    
    # Translate summary to Hindi
    mean = translate(summary, 'hindi')
    print(f"Translation: {mean}")
    stream_data(mean,0.02)
    
    # Generate image prompts
    i = 0
    while i < 4:
        imgPrompt = prompt(summary)
        prompt_list.append(imgPrompt)
        print(f'Prompt for image: {imgPrompt}')
        i += 1
    
    # Display generated images
    with st.spinner("Visualizing..."):
        for prom in prompt_list:
            img, f = IG(f'3D realistic cartoonistic ultra HD {prom}')
            st.image(img)

def upload_image():

    cap = cv2.VideoCapture(1) 

    if not cap.isOpened():
        print("Error: Could not open video device")
        exit()

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    file_path = os.path.join('uploaded_images', 'received_image.png')

    while True:
        ret, frame = cap.read()

        if ret:
            cv2.imshow("Camera Feed", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite(file_path, frame)
                print("Image captured and saved as 'received_image.png'.")
                break
        else:
            print("Error: Failed to capture image")
            break

    cap.release()
    cv2.destroyAllWindows()

    # Call the processing function to process the image
    file = Image.open(file_path)
    processing(file)

# Streamlit UI
st.title("Text Visualizer")
st.divider()

# Sidebar for file upload
sidebar = st.sidebar
sidebar.title("Menu")
file = sidebar.file_uploader("Choose a file")

# Tabs for different sections
home, about = st.tabs(["Home", "About"])

with home:
    if file:
        # Open the uploaded file with PIL
        file = Image.open(file)
        placeholder = sidebar.empty()
        placeholder.success("File Uploaded successfully")
        time.sleep(1)
        placeholder.empty()

        home.image(file, caption='Uploaded File.', width=500)
        
        with st.spinner("Thinking...."):
            processing(file)

with about:
    left, right = st.columns([1,2])
    with left:
        st.markdown("""
        <h5>Name</h5><br>
        <h5>Email</h5><br>
        <h5>Description</h5><br>
        <h5>Other links</h5><br>            
""", unsafe_allow_html=True)
    with right:
        st.markdown("""
        <h6>Ritesh And Group</h6><br>
        <h6>karanstdio1234@gmail.com</h6><br>
        <h6>AI Web App for Visualizing content in pages</h6><br>
        <h6>https://www.linkedin.com/in/ritesh-pandit-408557269/</h6><br>   
        <h6>https://github.com/</h6>         
""", unsafe_allow_html=True)
    st.divider()

    # Generate the markdown content with improved CSS styling
    about_text = f"""
    <div style="background-color: rgb(40 40 40); padding: 20px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); font-family: Arial, sans-serif; color: white;">
        <h2 style="font-size: 24px; color: white; margin-bottom: 15px;">About This Virtual Assistant</h2>
        <p>This virtual assistant, built using Streamlit, is capable of performing various tasks to assist you with your daily activities and queries. Below are some of its capabilities:</p>

        <ul style="list-style-type: disc; padding-left: 20px;">
    """
    about_text += """
        </ul>
    </div>
    """

    # Display the styled markdown in the About tab
    st.markdown(about_text, unsafe_allow_html=True)
