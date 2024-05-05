###     REQUIREMENTS    ###
##create virtual environment
#py -3.10 -m venv .venv

##to run the code (run command in cmd)
#streamlit run nlpproject.py

##library installation
#pip install torch
#pip install transformers
#pip install streamlit
#pip install bs4
#pip install pandas
#pip install textblob

from textblob import TextBlob
import pandas as pd 
import streamlit as st

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import requests
from bs4 import BeautifulSoup
import re
import numpy as np

#instantaite model
tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')


st.header('SENTIMENT ANALYSIS.')

#for text from user
with st.expander('Analyze Text'):
    text = st.text_input('Type text here: ')
    if text:
        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)
        st.write('Rating (*/5) : ',int(torch.argmax(result.logits))+1)

#retrieve the url of Yelp website
with st.expander('YELP REVIEW'):
    text = st.text_input('Give the link here: ')
    if text:
        #collect review from the Yelp website
        r = requests.get(text)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*comment.*')
        results = soup.find_all('p', {'class':regex})
        reviews = [result.text for result in results]

        df = pd.DataFrame(np.array(reviews), columns=['review'])

        #encode and calculate sentiment
        def sentiment_score(review):
            tokens = tokenizer.encode(review, return_tensors='pt')
            result = model(tokens)
            return int(torch.argmax(result.logits))+1

        #adding rating column to dataframe
        df['Rating (*/5)'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

        #displaying the dataframe to user
        df

#retrieve the Foursquare website
with st.expander('FOURSQUARE REVIEW'):
    ##collect review from the Foursquare website
    text = st.text_input('Link here: ')
    if text:
        r = requests.get(text)
        soup = BeautifulSoup(r.text, 'html.parser')
        regex = re.compile('.*tipText.*')
        results = soup.find_all('div', {'class':regex})
        reviews = [result.text for result in results]

        df = pd.DataFrame(np.array(reviews), columns=['review'])

        #encode and calculate sentiment
        def sentiment_score(review):
            tokens = tokenizer.encode(review, return_tensors='pt')
            result = model(tokens)
            return int(torch.argmax(result.logits))+1
        
        #adding rating column to dataframe
        df['Rating (*/5)'] = df['review'].apply(lambda x: sentiment_score(x[:512]))

        #displaying the dataframe to user
        df
