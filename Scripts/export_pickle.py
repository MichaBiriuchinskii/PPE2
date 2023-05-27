import pickle
from datastructures import Corpus, Article

# extension pickle : .pkl

def write_pickle(corpus:Corpus, destination:str):
    with open(destination, "wb") as fout:
        pickle.dump(corpus, fout)
