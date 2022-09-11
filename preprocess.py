from gensim.utils import tokenize
from pymorphy2.analyzer import MorphAnalyzer
from nltk.corpus import stopwords
from tqdm import tqdm
import nltk

class Preprocessor:
    nltk.download("stopwords")
    russian_stopwords = stopwords.words("russian")
    morph = MorphAnalyzer()
    def lemmatize(text): 
        res = list()
        for word in text:
            p = morph.parse(word)[0]
            res.append(p.normal_form)
        return res

    def tokenization(text):
        clean=[]
        for line in tqdm(text):
            line_2=lemmatize(list(tokenize(line, lowercase=True, deacc = True)))
            line_2=[i for i in line_2 if len(i)>1]
            clean.append(line_2)
        clean = [i for i in clean if i != []]
        return clean

    def vectorize(text):
        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=russian_stopwords, lowercase=False, use_idf=True, min_df=0)    
        tfidf.fit(clean)
        return tfidf
    def pred_preprocess(text):
