from gensim.models import Word2Vec
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pickle



class W2V:
    def __init__(self, vector_size=16, window=4):
        self.vector_size = vector_size
        self.window = window    
         
    def fit(self, text):
        self.word2vec = Word2Vec(min_count=1,
                    window=self.window,
                    size=self.vector_size, 
                    alpha=0.03, 
                    min_alpha=0.0007, 
                    sg = 1,
                    workers=4)
        self.word2vec.build_vocab(text, progress_per=10000)
        self.word2vec.train(text, total_examples=len(text), epochs=5, report_delay=1)
        return self
    def __getitem__(self, line):
        return self.word2vec.wv[line]

    def get_vocab(self):
        return self.word2vec.wv.vocab

    def __call__(self):
        return self.get_embeddings(line)



class Generator:
    def __init__(self, w2v):
        self.w2v = w2v

    def ged_dict(text):
        dct = Dictionary(clean) 
        dict_py = dct.token2id
        self.dct = dct
        self.dict_py = dict_py

    def transform(n_last, text):
        get_dict(text)
        def tfidf_emb(line, text, ind):
            outp=[]
            keys = tfidf.get_feature_names_out()
            get_id = tfidf.transform(text).toarray()[ind]
            indexes = np.where(get_id > 0)[0]
            keys = keys[indexes]
            get_id = get_id[indexes]
            embedding_dict = dict()
            for i in range(len(keys)):
                embedding_dict[keys[i]] = get_id[i]
                embedding = []
                for word in line:
                    if(word in embedding_dict):
                        embedding.append(embedding_dict[word])
                    else:
                        embedding.append(0)
                outp=embedding
            return outp
        x=[]
        y=[]
        for line in tqdm(text):
            ind = clean.index(line)
            line = [word for word in line if word in self.w2v.gev_vocab()]
            embed = np.zeros_like((1,1))
            tfidf_line=tfidf_emb(line,text,ind)
            if(tfidf_line!=[]):
                embed = (np.array(self.w2v[line]).T*np.array(tfidf_line)).T
                if(len(line)>=2):
                    for i in range(len(line)-((len(line)//2)+1)):
                        offset = (len(line)//2)+i
                        temp = embed[:offset].mean(axis=0)
                        for j in range(1,n_last+1):
                            i_s =  embed[:offset][-j]
                            temp = np.concatenate((temp, i_s))
                            y_temp = self.dict_py[line[(len(line)//2)+i+1]]
                        x.append(temp)
                        y.append(y_temp)
        x = np.array(x)
        y = np.array(y)
        return (x,y)
    
    def xgb_train(x, y, train_size):
        xgb = XGBClassifier(
            n_estimators=70,
            learning_rate=0.5,
            max_depth=3,
            objective='multi:softmax'
            num_class = len(self.dict_py)
        )
        xgb.fit(x[:train_size+1],y[:train_size+1])
        self.xgb = xgb
        return xgb

    def predict(self, inp_str, length, rand_num,n_last_pred,num_last_words):
        num_words = len(inp_str)-1+length
        while len(inp_str) < num_words:
            temp=[]
            x_pred = []
            for word in inp_str:
                if word in w2v_model.wv.vocab:   
                    temp.append(w2v_model[word]) 
            x_pred = np.array(temp).mean(axis=0)
            for j in range(1, min(n_last_pred+1,len(inp_str))):
                x_pred = np.concatenate((x_pred, temp[-j]))
            x_pred = np.array([x_pred])
            y_pred = cat.predict(x_pred)
            pred_word = dct.get(y_pred[0][0])
            passed = True
            for i in range(1,min(num_last_words+1,len(inp_str))):
                if pred_word == inp_str[-i]:
                    passed=False
                if passed==False:
                        pred_word=w2v_model.most_similar(pred_word)[random.randint(1, rand_num-1)][0]
                        i-=1    
            inp_str.append(pred_word)
        return (' '.join(inp_str))

    def save_model(self, path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "wb") as f:
            pickle.dump(self, file=f)

    @staticmethod
    def load_model(path):
        if not path.endswith("pkl"):
            raise ValueError("Extension must be pkl")
        with open(f"{path}", "rb") as f:
            model = pickle.load(file=f)
        return model