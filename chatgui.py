import nltk
from nltk.stem import WordNetLemmatizer

import pickle
import numpy as np

from keras.models import load_model

import json
import random

from tkinter import *

# Lemmatize using WordNet's built-in morphy function.
# Returns the input word unchanged if it cannot be found in WordNet. ('dogs' -> 'dog', 'abaci' -> 'abacus' etc)
lemmatizer = WordNetLemmatizer()

# Load a pretrained model (.h5 model)
model_h5 = load_model('chatbot_model.h5')

# data file which has predefined patterns and responses.
intents = json.loads(open('intents.json').read())
# pickle file in which we store the words Python object that contains a list of our vocabulary.
words_pkl = pickle.load(open('words.pkl', 'rb'))
# pickle file contains the list of categories.
classes = pickle.load(open('classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    """
    Function that returns a list of shorter versions of the words in the msg
    :param sentence: msg written by the user
    :return: list of shorter versions of the words in the msg
    """
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    """
    Function that returns a bag of words array: 1 if word in the bag exists in the sentence, 0 otherwise
    :param sentence: msg written by the user
    :param words: pickle file
    :param show_details:
    :return: return a bag of words array
    """
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  # create a bag with N zeros
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words_pkl, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    result = None
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model_h5)
    res = getResponse(ints, intents)
    return res

# Graphic User Interface -----------------------------------------------------
# Creating GUI with tkinter
def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#8a1c1c", font=("Arial", 13))
    
        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')
        ChatLog.config(foreground="#0e8038", font=("Arial", 13))
            
        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)
 

base = Tk()
base.title("Hello")
base.geometry("400x500")
base.resizable(width=FALSE, height=FALSE)

# Create Chat window
ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

ChatLog.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

# Create Button to send message
SendButton = Button(base, font=("Arial", 12, 'bold'), text="Send", width="11", height=5,
                    bd=0, bg="#37bcd4", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=0, bg="white", width="29", height="5", font="Arial")
# EntryBox.bind("<Return>", send)


# Place all components on the screen
scrollbar.place(x=376, y=6, height=386)
ChatLog.place(x=6, y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

base.mainloop()
