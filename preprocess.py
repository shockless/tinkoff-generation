import nltk
from gensim.utils import tokenize
from nltk.corpus import stopwords
from pymorphy2.analyzer import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

morph = MorphAnalyzer()


def lemmatize(text):
    res = list()
    for word in text:
        p = morph.parse(word)[0]
        res.append(p.normal_form)
    return res


class Preprocessor:
    def __init__(self):
        nltk.download("stopwords")
        self.russian_stopwords = stopwords.words("russian")

    def tokenization(self, text):
        clean = []
        for line in tqdm(text):
            line_2 = lemmatize(list(tokenize(line, lowercase=True, deacc=True)))
            line_2 = [i for i in line_2 if len(i) > 1]
            clean.append(line_2)
        clean = [i for i in clean if i != []]
        return clean

    def vectorize(self, text):
        def identity_tokenizer(text):
            return text


        tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, stop_words=self.russian_stopwords, lowercase=False,
                                use_idf=True, min_df=0)
        tfidf.fit(text)
        return tfidf

    def pred_preprocess(self, text):
        return lemmatize(list(tokenize(text, lowercase=True, deacc=True)))
