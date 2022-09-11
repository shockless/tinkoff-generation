import argparse

from model import W2V, Generator
from preprocess import Preprocessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating')
    parser.add_argument('--model', type=str, help='Path to load the model')
    parser.add_argument('--prefix', default=None, type=str, help='Input to model')
    parser.add_argument('--length', default=1, type=int,  help='Textgen length')
    arguments = parser.parse_args()
    text = None
    if arguments.prefix:
        text=Preprocessor.pred_preprocess(arguments.prefix)

    gen = Generator.load_model(arguments.model)
    print(gen.predict(text, length, 3, 1, 4))
