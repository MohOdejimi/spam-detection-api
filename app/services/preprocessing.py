import nltk 
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 

nltk.download('punkt')
nltk.download('stopwords')

def remove_stop_words(tokens):
    stopwrds = set(stopwords.words('english'))
    filtered = [word for word in tokens if word not in stopwrds]
    return filtered

def concatenate_words(tokens):
    concat = ' '.join(tokens)
    return concat

def preprocess_message(message):
    tokens = word_tokenize(message.lower())
    return concatenate_words(remove_stop_words(tokens))