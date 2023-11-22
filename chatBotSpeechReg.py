
import streamlit as st
import speech_recognition as sr
import pyttsx3
import time
import pandas as pd
import nltk
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity      
import warnings
warnings.filterwarnings('ignore')
lemmatizer = nltk.stem.WordNetLemmatizer()

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')


data = pd.read_csv('Mental_Health_FAQ (2).csv')
data.drop('Question_ID', axis = 1, inplace = True)
# data

# --------------------------------------- CHATBOT IMPLEMENTATION -----------------------------

# Define a function for text preprocessing (including lemmatization)
def preprocess_text(text):
    global tokens
    # Identifies all sentences in the data
    sentences = nltk.sent_tokenize(text)
    
    # Tokenize and lemmatize each word in each sentence
    preprocessed_sentences = []
    for sentence in sentences:
        tokens = [lemmatizer.lemmatize(word.lower()) for word in nltk.word_tokenize(sentence) if word.isalnum()]
        # Turns to basic root - each word in the tokenized word found in the tokenized sentence - if they are all alphanumeric 
        # The code above does the following:
        # Identifies every word in the sentence 
        # Turns it to a lower case 
        # Lemmatizes it if the word is alphanumeric

        preprocessed_sentence = ' '.join(tokens)
        preprocessed_sentences.append(preprocessed_sentence)
    
    return ' '.join(preprocessed_sentences)


data['tokenized Questions'] = data['Questions'].apply(preprocess_text)
# data.head(20)


corpus = data['tokenized Questions'].to_list()
# corpus

tfidf_vector = TfidfVectorizer()
v_corpus = tfidf_vector.fit_transform(corpus)


chatbot_greeting = [
    "Hello there, welcome to Orpheus Bot. pls ejoy your usage",
    "Hi user, This bot is created by oprheus, enjoy your usage",
    "Hi hi, How you dey my nigga",
    "Alaye mi, Abeg enjoy your usage",
    "Hey Hey, pls enjoy your usage"    
]

user_greeting = ["hi", "hello there", "hey", "hi there"]
exit_word = ['bye', 'thanks bye', 'exit', 'goodbye']

def bot_response(user_input):
    user_input_processed = preprocess_text(user_input)
    v_input = tfidf_vector.transform([user_input_processed])
    most_similar = cosine_similarity(v_input, v_corpus)
    most_similar_index = most_similar.argmax()
    
    return data['Answers'].iloc[most_similar_index]



# -------------------------------------SPEECH RECOGNITION IMPLEMENTATION  ---------------------------------
def transcribe_speech():
    # Initialize recognizer class
    r = sr.Recognizer()

    # Reading Microphone as source
    with sr.Microphone() as source:

        # create a streamlit spinner that shows progress
        with st.spinner(text='Silence pls, Caliberating background noise.....'):
            time.sleep(2)

        r.adjust_for_ambient_noise(source, duration = 1) # ..... Adjust the sorround noise
        st.info("Speak now...")
        global text
        audio_text = r.listen(source) #........................ listen for speech and store in audio_text variable
        with st.spinner(text='Transcribing your voice to text'):
            time.sleep(2)
        global text     
        try:
            # using Google speech recognition to recognise the audio
            text = r.recognize_google(audio_text)
            # print(f' Did you say {text} ?')
            return text
        except:
            return "Sorry, I did not get that."


# ----------------------------------- STREAMLIT DESIGN -----------------------

st.markdown("<h1 style = 'text-align: center; color: #0F0F0F'>MENTAL HEALTH CHATBOT</h1>", unsafe_allow_html = True)

st.markdown("<br> <br>", unsafe_allow_html= True)


st.sidebar.image('pngwing.com (4).png',width=300, caption = 'Mental Health Related Chats')

option = st.sidebar.selectbox("Select an option", ["Home", "Speech Recognition"])

if option == "Home":
    st.image('pngwing.com (5).png',width=300, caption = 'Mental Health Background')
    st.markdown("<h3 style = 'color: #0F0F0F; text-align: center;font-family: Arial, Helvetica, sans-serif; '>BACKGROUND OF  THE PROJECT </h2>", unsafe_allow_html= True)
    # st.markdown('<br><br>', unsafe_allow_html= True)
    st.markdown("<p>Mental health is a spectrum that encompasses various conditions and challenges. It is essential to recognize that struggling with mental health does not diminish one's strength or worth. Mental health conditions are common, and seeking help is a sign of courage and resilience..</p>",unsafe_allow_html= True)
    st.write("Welcome to the Mental Health Speech Recognition Chatbot!")
elif option == "Speech Recognition":
    st.image('pngwing.com (6).png',width=200)
    if st.button("Start Recording"):
        st.write("Click on the microphone to start speaking:")
        your_words_in_text = transcribe_speech()
        st.write("Transcription: ", your_words_in_text)

        user_q = your_words_in_text
        if user_q in user_greeting:
            st.write(random.choice(chatbot_greeting))
        elif user_q in exit_word:
            st.write('Thank you for your usage. Bye')
        elif user_q == '':
            st.write('Pls ask your question')
        else:
            responses = bot_response(user_q)
            st.success(f'ChatBot:  {responses}')

st.markdown('<br><br>', unsafe_allow_html= True)
st.markdown("<h5 style = 'text-align: center; top-margin: 0rem; color: #164863'>BUILT BY JAYEOBA OLAMILEKAN</h1>", unsafe_allow_html = True)

