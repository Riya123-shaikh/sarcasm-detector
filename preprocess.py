# preprocess.py
import re
import string
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def clean_tweet(text):
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower()
    tokens = [word for word in text.split() if word not in STOPWORDS and len(word) > 2]
    return " ".join(tokens)
