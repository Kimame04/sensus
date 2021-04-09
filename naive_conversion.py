import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet

def test(string,isSubjective):
    nltk.download('stopwords')
    nltk.download('wordnet')
    strings = string.split(' ')
    stop_words = set(stopwords.words('english')) 
    res = ''
    for string in strings: 
        if string in stop_words or not string.islower(): 
            temp = ' ' + string + ' '
            res+=temp
            continue
        if len(wordnet.synsets(string)) != 0:
            temp = ' ' + find_synonyms(string,isSubjective)[0][1] + ' '
        else: temp = string + ' '
        res += temp 
    regex = re.compile(r"\s+")
    res = regex.sub(" ", res).strip()
    return res

def find_synonyms(word,isSubjective):
    model_direct = pickle.load(open('objectivity-detection-direct.sav','rb'))
    list_synonyms = []
    for syn in wordnet.synsets(word):
        for lemm in syn.lemmas():
            list_synonyms.append(lemm.name())
    #list_synonyms = [item.lower() for item in list_synonyms]
    scores = [(model_direct.predict_proba([text])[0][0],text) for text in list_synonyms]
    scores.append((model_direct.predict_proba([word])[0][0],word))
    if isSubjective:
        scores.sort(reverse=True)
    else: scores.sort()
    return scores