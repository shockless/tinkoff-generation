import numpy as np
import argparse
import os

from preprocess import Preprocessor
from model import W2V, Generator

if __name__ == "__main__":
    import nltk
    nltk.download('stopwords')

    parser = argparse.ArgumentParser(description='Train and fit')
    parser.add_argument('--input-dir', type=str, help='Dir data path')
    parser.add_argument('--model', type=str, help='Model save path')
    arguments=parser.parse_args()
    lines=[]
    if arguments.input_dir:
        for file in os.listdir(arguments.input_dir):
            file = open(f"{arguments.input_dir}/{file}", "r", encoding="utf8")
            line=file.readlines()
            for lin in line:
                lines.append(lin)
    else:
        lines=str(input())
    lines=[]
    if arguments.input_dir:
        for file in os.listdir(arguments.input_dir):
            file = open(f"{arguments.input_dir}/{file}", "r", encoding="utf8")
            line=file.readlines()
            for lin in line:
                lines.append(lin)
    else:
        lines=str(input())
    clean = []

    p = Preprocessor()
    clean = p.tokenization(lines)
    w2v = W2V()

    w2v.fit(clean)
    generator = Generator(w2v=w2v)
    x,y = generator.transform(1, clean)
    generator.xgb_train(x,y,5000)
    generator.save_model(arguments.model)