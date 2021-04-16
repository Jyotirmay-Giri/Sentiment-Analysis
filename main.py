import datetime as dt
import re

import pandas as pd
import streamlit as st
from flair.data import Sentence
from flair.models import TextClassifier


# Set page title
st.title('Amazon Sentiment Analysis')

# Load classification model
with st.spinner('Loading classification model...'):
    classifier = TextClassifier.load('models/best-model.pt')

# Preprocess function
allowed_chars = ' AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz0123456789~`!@#$%^&*()-=_+[]{}|;:",./<>?'
punct = '!?,.@#'
maxlen = 280

def preprocess(text):
    # Delete URLs, cut to maxlen, space out punction with spaces, and remove unallowed chars
    return ''.join([' ' + char + ' ' if char in punct else char for char in [char for char in re.sub(r'http\S+', 'http', text, flags=re.MULTILINE) if char in allowed_chars]])

### SINGLE REVIEW CLASSIFICATION ###
st.subheader('Single review classification')

# Get sentence input, preprocess it, and convert to flair.data.Sentence format
review_input = st.text_input('Review:')

if review_input != '':
    # Pre-process review
    sentence = Sentence(preprocess(review_input))

    # Make predictions
    with st.spinner('Predicting...'):
        classifier.predict(sentence)

    # Show predictions
    label_dict = {'0': 'Negative', '4': 'Positive'}

    if len(sentence.labels) > 0:
        st.write('Prediction:')
        st.write(label_dict[sentence.labels[0].value] + ' with ',
                sentence.labels[0].score*100, '% confidence')

### REVIEW SEARCH AND CLASSIFY ###
st.subheader('Search review for Query')

# Get user input
query = st.text_input('Query:', '#')

# As long as the query is valid (not empty or equal to '#')...
if query != '' and query != '#':
    with st.spinner(f'Searching for and analyzing {query}...'):
        # Get English review from the past 4 weeks
        reviews = query_reviews(query, begindate=dt.date.today() - dt.timedelta(weeks=4), lang='en')

        # Initialize empty dataframe
        review_data = pd.DataFrame({
            'review': [],
            'predicted-sentiment': []
        })

        # Keep track of positive vs. negative review
        pos_vs_neg = {'0': 0, '4': 0}

        # Add data for each tweet
        for review in review:
            # Skip iteration if review is empty
            if review.text in ('', ' '):
                continue
            # Make predictions
            sentence = Sentence(preprocess(review.text))
            classifier.predict(sentence)
            sentiment = sentence.labels[0]
            # Keep track of positive vs. negative reviews
            pos_vs_neg[sentiment.value] += 1
            # Append new data
            review_data = tweet_data.append({'tweet': review.text, 'predicted-sentiment': sentiment}, ignore_index=True)

# Show query data and sentiment if available
try:
    st.write(review_data)
    try:
        st.write('Positive to negative tweet ratio:', pos_vs_neg['4']/pos_vs_neg['0'])
    except ZeroDivisionError: # if no negative tweets
        st.write('All postive tweets')
except NameError: # if no queries have been made yet
    pass