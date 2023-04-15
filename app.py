import nltk
from nltk.corpus import stopwords
import pickle
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


@st.experimental_singleton
def load_stop_words():
    nltk.download('stopwords')
    return set(stopwords.words('english'))


@st.experimental_singleton
def load_model():
    return tf.keras.models.load_model('best_model.h5')


@st.experimental_singleton
def load_tokernizer():
    with open('tokenizer.pickle', 'rb') as f:
        out = pickle.load(f)
    return out


MAX_LENGTH = 300
TRUNC_TYPE = 'post'

tokenizer = load_tokernizer()
stop_words = load_stop_words()
model = load_model()


def remove_stop_words(s):
    s = s.lower()
    return [" ".join([w for w in s.split() if w not in stop_words])]


def predict_prob(text):
    tokens = tokenizer.texts_to_sequences(remove_stop_words(text))
    pad_seq = pad_sequences(tokens, maxlen=MAX_LENGTH, truncating=TRUNC_TYPE)
    return model.predict(pad_seq, verbose=0)[0][0]


st.write("""
# Break Point
### Detect suicidal thoughts
""")

text = st.text_input('Enter a thought below')
if text:
    suicidal_prob = predict_prob(text)
    suicidal = "not"
    if suicidal_prob >= 0.9:
        suicidal = ""
    st.write(f'**Thought is {suicidal} Suicidal.**')

st.write("""
## Project Goal
Build a tool to detect suicidal thoughts

## Methodology
Develop a binary text classification model using labelled text blobs of suicidal and non-suicidal thoughts

## Data
Data for model training was downloaded from Kaggle; ([Suicide and Depression Detection](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)).
Entire dataset has 116,037 instances each in non-suicide and suicide category. Text from suicide category (654 words at 
50th percentile) was lengthier than non-suicide(166 words at 50th percentile). Dataset was split into a training(90% of 
the data) and a test(10% fo the data) set. Training dataset had 190,576 unique words, however only 10% of the words had 
18 or more occurrences (which helps to decide the size of the vocabulary for the tokenizer).   
""")

st.write("""
## Model training
1. Tensorflow was used as the model training library
2. Dev set was used to identify model hyper-parameters efficiently by following a greedy approach
3. Since the dataset was balanced, accuracy was used to compare the model performance
4. Along with identified hyper-parameters, text with stop words removed had relatively better performance on the dev set
5. These configurations were, then used to train the final Bi-directional LSTM model
6. Test accuracy (94.7%) peaked at the 8th epoch
7. Looking at the precision and recall curve for the best model, a threshold of 0.9 on prediction probability provides 95% precision and 90% recall (Precision was deemed more important to avoid unnecessary panic)
8. Used chatGPT to generate suicidal, non-suicidal and neutral thoughts (5 thoughts for category), model performed with 100% accuracy on these text with the 0.9 threshold  

## Explore the code
[Break-Point](https://github.com/nis89mad/Break-Point)
""")
