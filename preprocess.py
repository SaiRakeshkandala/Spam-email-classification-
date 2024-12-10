import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

def preprocess_text(text):
    # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z]', ' ', text)
            # Tokenize and remove stopwords
                tokens = word_tokenize(text.lower())
                    stop_words = set(stopwords.words('english'))
                        tokens = [word for word in tokens if word not in stop_words]
                            # Stemming
                                stemmer = PorterStemmer()
                                    tokens = [stemmer.stem(word) for word in tokens]
                                        return ' '.join(tokens)
                                        