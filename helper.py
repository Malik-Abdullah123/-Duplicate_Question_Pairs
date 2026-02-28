import numpy as np
import distance
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup


def preprocess(q):
    q = str(q).lower().strip()

    # Replace certain special characters with their string equivalents
    q = q.replace('%', ' percent')
    q = q.replace('$', ' dollar ')
    q = q.replace('₹', ' rupee ')
    q = q.replace('€', ' euro ')
    q = q.replace('@', ' at ')

    # The pattern '[math]' appears around 900 times in the whole dataset.
    q = q.replace('[math]', '')

    # Replacing some numbers with string equivalents (not perfect, can be done better to account for more cases)
    q = q.replace(',000,000,000 ', 'b ')
    q = q.replace(',000,000 ', 'm ')
    q = q.replace(',000 ', 'k ')
    q = re.sub(r'([0-9]+)000000000', r'\1b', q)
    q = re.sub(r'([0-9]+)000000', r'\1m', q)
    q = re.sub(r'([0-9]+)000', r'\1k', q)

    # Decontracting words
    # https://en.wikipedia.org/wiki/Wikipedia%3aList_of_English_contractions
    # https://stackoverflow.com/a/19794953
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "can not",
        "can't've": "can not have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he'll've": "he will have",
        "he's": "he is",
        "how'd": "how did",
        "how'd'y": "how do you",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'd've": "i would have",
        "i'll": "i will",
        "i'll've": "i will have",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'd've": "it would have",
        "it'll": "it will",
        "it'll've": "it will have",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "mightn't've": "might not have",
        "must've": "must have",
        "mustn't": "must not",
        "mustn't've": "must not have",
        "needn't": "need not",
        "needn't've": "need not have",
        "o'clock": "of the clock",
        "oughtn't": "ought not",
        "oughtn't've": "ought not have",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "shan't've": "shall not have",
        "she'd": "she would",
        "she'd've": "she would have",
        "she'll": "she will",
        "she'll've": "she will have",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "shouldn't've": "should not have",
        "so've": "so have",
        "so's": "so as",
        "that'd": "that would",
        "that'd've": "that would have",
        "that's": "that is",
        "there'd": "there would",
        "there'd've": "there would have",
        "there's": "there is",
        "they'd": "they would",
        "they'd've": "they would have",
        "they'll": "they will",
        "they'll've": "they will have",
        "they're": "they are",
        "they've": "they have",
        "to've": "to have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'd've": "we would have",
        "we'll": "we will",
        "we'll've": "we will have",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what'll've": "what will have",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "when's": "when is",
        "when've": "when have",
        "where'd": "where did",
        "where's": "where is",
        "where've": "where have",
        "who'll": "who will",
        "who'll've": "who will have",
        "who's": "who is",
        "who've": "who have",
        "why's": "why is",
        "why've": "why have",
        "will've": "will have",
        "won't": "will not",
        "won't've": "will not have",
        "would've": "would have",
        "wouldn't": "would not",
        "wouldn't've": "would not have",
        "y'all": "you all",
        "y'all'd": "you all would",
        "y'all'd've": "you all would have",
        "y'all're": "you all are",
        "y'all've": "you all have",
        "you'd": "you would",
        "you'd've": "you would have",
        "you'll": "you will",
        "you'll've": "you will have",
        "you're": "you are",
        "you've": "you have"
    }

    q_decontracted = []

    for word in q.split():
        if word in contractions:
            word = contractions[word]

        q_decontracted.append(word)

    q = ' '.join(q_decontracted)
    q = q.replace("'ve", " have")
    q = q.replace("n't", " not")
    q = q.replace("'re", " are")
    q = q.replace("'ll", " will")

    # Removing HTML tags
    q = BeautifulSoup(q)
    q = q.get_text()

    # Remove punctuations
    pattern = re.compile('\W')
    q = re.sub(pattern, ' ', q).strip()

    return q
def common_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return len(w1 & w2)

def total_words(row):
    w1 = set(map(lambda word: word.lower().strip(), row['question1'].split(" ")))
    w2 = set(map(lambda word: word.lower().strip(), row['question2'].split(" ")))
    return (len(w1) + len(w2))



stop_word = set(stopwords.words('english'))

def fetch_token_features(row):

    q1 = re.sub(r'[^\w\s]', '', row['question1'].lower())
    q2 = re.sub(r'[^\w\s]', '', row['question2'].lower())

    safe_div = 0.0001
    token_features = [0.0] * 8

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = {w for w in q1_tokens if w not in stop_word}
    q2_words = {w for w in q2_tokens if w not in stop_word}

    q1_stop = {w for w in q1_tokens if w in stop_word}
    q2_stop = {w for w in q2_tokens if w in stop_word}

    common_word_count = len(q1_words & q2_words)
    common_stopword_count = len(q1_stop & q2_stop)
    common_tokens_count = len(set(q1_tokens) & set(q2_tokens))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + safe_div)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + safe_div)
    token_features[2] = common_stopword_count / (min(len(q1_stop), len(q2_stop)) + safe_div)
    token_features[3] = common_stopword_count / (max(len(q1_stop), len(q2_stop)) + safe_div)
    token_features[4] = common_tokens_count / (min(len(q1_tokens), len(q2_tokens)) + safe_div)
    token_features[5] = common_tokens_count / (max(len(q1_tokens), len(q2_tokens)) + safe_div)

    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features
import pickle

cv = pickle.load(open('cv.pkl','rb'))

def fetch_length_features(row):
    q1 = row['question1']
    q2 = row['question2']

    length_features = [0.0] * 3

    # Converting the Sentence into Tokens:
    q1_tokens = q1.lower().split()
    q2_tokens = q2.lower().split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Average Token Length of both Questions
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))
    length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)

    return length_features
def fetch_fuzz_features(row):
    q1 = row['question1']
    q2 = row['question2']

    fuzzy_features = [0.0]*4

    # fuzz_ratio
    fuzzy_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ratio
    fuzzy_features[1] = fuzz.partial_ratio(q1, q2)

    # token_sort_ratio
    fuzzy_features[2] = fuzz.token_sort_ratio(q1, q2)

    # token_set_ratio
    fuzzy_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzzy_features

def test_common_words(q1,q2):
    w1=set(map(lambda word:word.lower().strip(), q1.split(" ")))
    w2=set(map(lambda word:word.lower().strip(), q2.split(" ")))
    return len(w1 & w2)

def test_total_words(q1,q2):
    w1=set(map(lambda word:word.lower().strip(), q1.split(" ")))
    w2=set(map(lambda word:word.lower().strip(), q2.split(" ")))
    return (len(w1) + len(w2))


def test_token_features(q1, q2):
    SAFE_DIV = 0.001

    stop_words = set(stopwords.words('english'))

    token_features = [0.0] * 8

    import re
    q1 = re.sub(r'[^\w\s]', '', q1)
    q2 = re.sub(r'[^\w\s]', '', q2)

    # Converting the Sentence into Tokens:
    q1_tokens = q1.lower().split()
    q2_tokens = q2.lower().split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    # get the non_stopword in Question
    q1_words = set([word for word in q1_tokens if word not in stop_words])
    q2_words = set([word for word in q2_tokens if word not in stop_words])

    # get the stopwords in Question
    q1_stops = set([word for word in q1_tokens if word in stop_words])
    q2_stops = set([word for word in q2_tokens if word in stop_words])

    # get the common word
    common_word = len(q1_words.intersection(q2_words))

    # get stop word in Question
    common_stop = len(q1_stops.intersection(q2_stops))

    # get common tokens in question
    common_tokens = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_tokens / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_tokens / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)

    # last word of both question is same or not
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])

    # first word same or not
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])

    return token_features


def test_length_features(q1, q2):
    length_features = [0.0] * 3

    q1_tokens = q1.lower().split()
    q2_tokens = q2.lower().split()

    import re
    q1 = re.sub(r'[^\w\s]', '', q1)
    q2 = re.sub(r'[^\w\s]', '', q2)

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return length_features

    # Absolute length features
    length_features[0] = abs(len(q1_tokens) - len(q2_tokens))

    # Avg token length
    length_features[1] = (len(q1_tokens) + len(q2_tokens)) / 2

    strs = list(distance.lcsubstrings(q1, q2))

    if len(strs) == 0:
        length_features[2] = 0
    else:
        length_features[2] = len(strs[0]) / (min(len(q1), len(q2)) + 1)
    return length_features


def test_fuzz_features(q1, q2):
    fuzz_features = [0.0] * 4

    # fuzz_ratio
    fuzz_features[0] = fuzz.QRatio(q1, q2)

    # fuzz_partial_ration
    fuzz_features[1] = fuzz.partial_ratio(q1, q2)

    # fuzz_sort_ratio
    fuzz_features[2] = fuzz.token_sort_ratio(q1, q2)

    # fuzz set ratio
    fuzz_features[3] = fuzz.token_set_ratio(q1, q2)

    return fuzz_features


def query_point_creator(q1, q2):
    input_query = []

    # preprocess
    q1 = preprocess(q1)
    q2 = preprocess(q2)

    # fetch basic features
    input_query.append(len(q1))
    input_query.append(len(q2))

    input_query.append(len(q1.split(" ")))
    input_query.append(len(q2.split(" ")))

    input_query.append(test_common_words(q1, q2))
    input_query.append(test_total_words(q1, q2))
    input_query.append(round(test_common_words(q1, q2) / test_total_words(q1, q2), 2))

    # fetch token features
    token_features = test_token_features(q1, q2)
    input_query.extend(token_features)

    # fetch length based features
    length_features = test_length_features(q1, q2)
    input_query.extend(length_features)

    # fetch fuzzy features
    fuzzy_features = test_fuzz_features(q1, q2)
    input_query.extend(fuzzy_features)

    # bow feature for q1
    q1_bow = cv.transform([q1]).toarray()

    # bow feature for q2
    q2_bow = cv.transform([q2]).toarray()

    return np.hstack((np.array(input_query).reshape(1, 22), q1_bow, q2_bow))
