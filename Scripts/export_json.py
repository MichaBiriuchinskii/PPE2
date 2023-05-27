import json
from dataclasses import asdict

from datastructures import Corpus, Article


def write_json(corpus:Corpus, destination:str):
    with open(destination, "w") as fout:
        json.dump(asdict(corpus), fout)

