import cv2
import os
import pytesseract
import numpy as np
import google.generativeai as genai
import streamlit as st
import time
from PIL import Image
from otherImgGen import ImageGenerator as IG
import PyPDF2


# Configure API key and initialize model
genai.configure(api_key="")
model = genai.GenerativeModel('gemini-1.5-flash')

def stream_data(data, delay: float = 0.1):
    placeholder = st.empty()  
    text = ""
    for word in data.split():
        text += word + " "
        placeholder.markdown(f"""<p style="color:rgb(27, 233, 212)">{text}</p>""", unsafe_allow_html= True) 
        time.sleep(delay)


def recognize_text(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, lang='hin+eng')            # lang='hin+eng'
        return text.strip()
    except Exception as e:
        st.error(e)
        return None

def ans(ask):
    try:
        response = model.generate_content(f"""explain {ask}. and write everything                               
                in html format with inline style so render in markdown of streamlit. 
                nothing else. don't use * symbols. use proper font and spaces. dont use ''' html in beginning""")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        st.error(e)

def summarize_text(text, exp):
    try:
        if exp:
            response = model.generate_content(f"""explain and solve if maths 
                                              problem and provide the calculation and 
                                              answer for {text}. and write everything                               
                in html format with inline style so render in markdown of streamlit. 
                nothing else. don't use * symbols. use proper font and spaces. dont use ''' html in beginning. If there
                is not mathematical expression then explain the text in detail""")
        else:
            response = model.generate_content(f"Summarize the following text: {text}")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        st.error(e)

def prompt(text):
    try:
        response = model.generate_content(f"Generate a prompt for image generation for the given text: {text}. Only generate the single best prompt and nothing else.")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        st.error(e)

def translate(summary, lang):
    try:
        response = model.generate_content(f"Translate the following {summary} into {lang} and use simple words.")
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        st.error(e)

def is_image(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return True
    except (IOError, SyntaxError):
        return False

def read_pdf(file):
    
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text += page.extract_text()
    return text

def processing(file, lang, flag, exp):
    prompt_list = []
    
    if flag:
        # Convert PIL image to NumPy array for OpenCV processing
        image = np.array(file)
        # Recognize text from the image
        texts = recognize_text(image)
        #print(f'Recognized texts: {texts}')
    else:
        texts = file
    # Summarize the recognized text
    summary = summarize_text(texts, exp)
    #print(f'Summary: {summary}')
    stream_data(summary, 0.02)
    
    # Translate summary to Hindi

    mean = translate(summary, lang)
    #print(f"Translation: {mean}")
    stream_data(mean,0.02)

    # Generate image prompts
    if exp==False:
        with st.spinner("Visualizing..."):
            i = 0
            while i < 4:
                imgPrompt = prompt(summary)
                prompt_list.append(imgPrompt)
                #print(f'Prompt for image: {imgPrompt}')
                i += 1
            
        # Display generated images
        
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
st.markdown("""<h1 style="font-style:italic;">Text <span style="color:orange;font-style:italic">Visualizer</span></h1>""",unsafe_allow_html=True)
st.divider()
flag = 0
# Sidebar for file upload
sidebar = st.sidebar
sidebar.title("Menu")
file = sidebar.file_uploader("Choose a file")
mainFile = file

# Tabs for different sections
home, tips, about = st.tabs(["Home", "Tips", "About"])

with home:
    
    st.markdown(f"""<h4 style="color:orange">You have to upload a file(image or PDF) to see the<span style="font-size:50px;color:white"> Magic</span>....</h3>""", unsafe_allow_html=True)
    ask = home.text_input("Ask Something...", placeholder="Key of Knowledge...")
    s1, s2 = st.columns(2)
    with s1:
        if st.button("ANS - SUMMARIZE"):
            if ask:
                res = ans(ask)
                stream_data(res)
    with s2:
        exp = False
        if st.button("EXPLAIN - SOLVE"):
            exp = True
    if file:
        # Open the uploaded file with PIL
        placeholder = sidebar.empty()
        if is_image(file):
            file = Image.open(file)
            placeholder.success("File Uploaded successfully")
            time.sleep(1)
            placeholder.empty()
            home.image(file, caption='Uploaded File.', width= 500)
            flag = 1
        else:
            file = read_pdf(file)
            placeholder.success("File Uploaded successfully")
            time.sleep(1)
            placeholder.empty()
            flag = 0
        if exp == False:
            lang = st.selectbox("Choose Language for Translation", ["None", "Hindi", "Marathi", "Tamil", "Telugu", "urdu", "Gujarati", "Other"])
            if(lang == 'Other'):
                lang = st.text_input("Enter Your preferred language")
            if(lang!='None' and lang!=''):
                with st.spinner("Thinking...."):
                    processing(file, lang, flag, exp)
        else:
            with st.spinner("Thinking...."):
                    lang = 'hindi'
                    processing(file, lang, flag, exp)

with tips:
    st.markdown(
    """
    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">1. Set Clear Goals</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Why you're studying/reading:</span> 
            Determine the purpose of your study or reading session. Are you preparing for an exam, seeking personal growth, or exploring new topics?
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Chapter or topic targets:</span> 
            Break down your tasks into manageable sections (e.g., one chapter, 20 pages, etc.).
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">2. Create a Study/Reading Schedule</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Regular intervals:</span> 
            Establish a routine by setting specific times for study and reading each day.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Chunked sessions:</span> 
            Study or read in short, focused intervals (25–50 minutes) followed by 5–10 minute breaks (Pomodoro technique).
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">3. Active Reading/Studying</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Take notes:</span> 
            Write summaries or key points as you read to reinforce learning.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Highlight selectively:</span> 
            Use highlighters or sticky notes for important concepts but avoid marking too much.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Ask questions:</span> 
            Engage with the material by asking yourself questions like, “How does this apply to real life?” or “What does this mean?”
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">4. Stay Organized</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Use bookmarks:</span> 
            Track your progress using bookmarks or digital tools.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Create outlines or mind maps:</span> 
            This can help you connect concepts and see the big picture.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Summarize regularly:</span> 
            After finishing a chapter or section, write a quick summary in your own words.
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">5. Focus on Comprehension, Not Speed</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Understand first:</span> 
            Focus on truly understanding the material instead of rushing through it. This will lead to better retention.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Re-read difficult sections:</span> 
            If something isn’t clear, re-read it. There’s no harm in slowing down when needed.
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">6. Eliminate Distractions</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Study environment:</span> 
            Choose a quiet, comfortable place for study sessions. Turn off notifications and minimize distractions.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Single-tasking:</span> 
            Focus on one task (one book or subject) at a time rather than multitasking.
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">7. Review and Recap</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Regular review:</span> 
            Revisit your notes and summaries to reinforce learning.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Teach someone else:</span> 
            Explaining concepts to someone else or writing them in your own words is a powerful way to deepen understanding.
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">8. Practice Active Recall</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Test yourself:</span> 
            Periodically quiz yourself on what you’ve read or studied without looking at the material. This helps reinforce memory.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Use flashcards:</span> 
            Digital tools like Anki can help with spaced repetition for long-term retention.
        </li>
    </ul>

    <h2 style="color: rgb(14, 176, 201); font-size: 24px; margin-bottom: 10px;">9. Vary Your Study Materials</h2>
    <ul style="list-style-type: none; padding-left: 10px;">
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Mix it up:</span> 
            Study or read from different sources like books, videos, and articles to get a well-rounded understanding.
        </li>
        <li style="margin-bottom: 15px;">
            <span style="color: orange; font-weight: bold;">Apply knowledge:</span> 
            Apply what you’ve learned in real-life situations or through practice exercises.
        </li>
    </ul>
    """,
    unsafe_allow_html=True
)


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
        st.markdown(f"""
        <h6 style="color:orange">Ritesh And Group</h6><br>
        <h6>karanstdio1234@gmail.com</h6><br>
        <h6 style="color:rgb(34, 246, 140)">AI Web App for Visualizing content in pages</h6><br>
        <h6>https://www.linkedin.com/in/ritesh-pandit-408557269/</h6><br>   
        <h6>https://github.com/</h6>         
""", unsafe_allow_html=True)
    st.divider()


