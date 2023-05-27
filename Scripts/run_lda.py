r"""
LDA Model
=========

Introduces Gensim's LDA model and demonstrates its use on the NIPS corpus.

"""

#import nltk
#nltk.download('wordnet')

import argparse, sys
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import io
import os.path
import re
import tarfile

import smart_open

def extract_documents(url='https://cs.nyu.edu/~roweis/data/nips12raw_str602.tgz'):
    with smart_open.open(url, "rb") as file:
        with tarfile.open(fileobj=file) as tar:
            for member in tar.getmembers():
                if member.isfile() and re.search(r'nipstxt/nips\d+/\d+\.txt', member.name):
                    member_bytes = tar.extractfile(member).read()
                    yield member_bytes.decode('utf-8', errors='replace')

#docs = list(extract_documents())

# Lemmatize the documents.
#from nltk.stem.wordnet import WordNetLemmatizer

#lemmatizer = WordNetLemmatizer()
#docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
print("DOCS GENSIM : ...")
#print(docs)

# docs est une liste de lemmes
from xml.etree import ElementTree as et

def conversion_xml(xml_file:str, lem_forme:str, cat_gram:list):
    """
        Prend en entrée :
            - un nom de fichier xml,
            - une chaîne pouvant prendre deux valeurs : 'lemme' ou 'forme',
            - une liste de catégories grammaticales à conserver dans l'analyse.
                si la liste est vide, on prend tous les tokens

        Produit une sortie la liste des lemmes (str) du document par article :

        [
            [art1_lemma1, art1_lemma2, …],
            [art2_lemma1, art2_lemma2, …],
            ...
        ]
    """
    output = []
    with open(xml_file, 'r') as f:
        contenu = f.read()
    # parser XML
    tree = et.parse(xml_file)
    root = tree.getroot()

    if len(cat_gram) == 0:
        # pour chaque article
        for article in root.iter('article'):
            tokens = []
            for token in article.iter('token'):
                lemme = token.get(lem_forme)
                if lemme is not None:
                    tokens.append(lemme)
            output.append(tokens)
    else:
        for article in root.iter('article'):
            tokens = []
            for token in article.iter('token'):
                lemme = token.get(lem_forme)
                # on filtre les catégories grammaticales
                categorie = token.get('pos')
                if lemme is not None and categorie in cat_gram:
                    tokens.append(lemme)
            output.append(tokens)
    return output

import pickle

def conversion_pkl(pkl_file:str, lem_forme:str, cat_gram:list):
    """
        Принимает на вход:
            - имя файла pkl,
            - строку, которая может принимать два значения: 'lemme' или 'forme',
            - список грамматических категорий для анализа.
              Если список пуст, выбираются все токены.

        Возвращает список лемм (str) для каждой статьи в документе:

        [
            [art1_lemma1, art1_lemma2, …],
            [art2_lemma1, art2_lemma2, …],
            ...
        ]
    """
    output = []
    with open(pkl_file, 'rb') as f:
        contenu = pickle.load(f)

    if len(cat_gram) == 0:
        # для каждой статьи
        for article in contenu.articles:
            tokens = []
            for token in article.analyse:
                if lem_forme == 'lemme':
                    lemme = token.lemme
                else:
                    lemme = token.forme
                if lemme is not None:
                    tokens.append(lemme)
            output.append(tokens)
    else:
        for article in contenu.articles:
            tokens = []
            for token in article.analyse:
                if lem_forme == 'lemme':
                    lemme = token.lemme
                else:
                    lemme = token.forme
                # фильтруем грамматические категории
                categorie = token.pos
                if lemme is not None and categorie in cat_gram:
                    tokens.append(lemme)
            output.append(tokens)
    return output

import json

def conversion_json(json_file: str, lem_forme: str, cat_gram: list):
    """
        Принимает на вход:
            - имя файла JSON,
            - строку, которая может принимать два значения: 'lemme' или 'forme',
            - список грамматических категорий для анализа.
              Если список пуст, выбираются все токены.

        Возвращает список лемм (str) для каждой статьи в документе:

        [
            [art1_lemma1, art1_lemma2, …],
            [art2_lemma1, art2_lemma2, …],
            ...
        ]
    """
    output = []
    with open(json_file, 'r') as f:
        contenu = json.load(f)

    if len(cat_gram) == 0:
        # для каждой статьи
        for article in contenu['articles']:
            tokens = []
            for token in article['analyse']:
                if lem_forme == 'lemme':
                    lemme = token['lemme']
                else:
                    lemme = token['forme']
                if lemme is not None:
                    tokens.append(lemme)
            output.append(tokens)
    else:
        for article in contenu['articles']:
            tokens = []
            for token in article['analyse']:
                if lem_forme == 'lemme':
                    lemme = token['lemme']
                else:
                    lemme = token['forme']
                # фильтруем грамматические категории
                categorie = token['pos']
                if lemme is not None and categorie in cat_gram:
                    tokens.append(lemme)
            output.append(tokens)
    return output

parser = argparse.ArgumentParser()
parser.add_argument("-f", help="format à lire (xml, json ou pkl)", default="xml")
parser.add_argument("-l", help="sélectionne le format (lemme ou forme)", default="lemme")
parser.add_argument("source_file", help="name of the file containing data to analyse")
parser.add_argument("categories", nargs="*", help="catégories grammaticales à retenir")
args = parser.parse_args()

if args.f == 'xml':
    conversion = conversion_xml
elif args.f == 'json':
    conversion = conversion_json
elif args.f == 'pkl':
    conversion = conversion_pkl
else:
    print("Méthode non disponible", file=sys.stderr)
    sys.exit()

# gestion des catégories grammaticales
cat_gram = args.categories

if args.l == 'lemme':
    docs = conversion(args.source_file, 'lemme', cat_gram)
elif args.l == 'forme':
    docs = conversion(args.source_file, 'forme', cat_gram)
else:
    print("Sélectionnez 'lemme' ou 'forme'", file=sys.stderr)
    sys.exit()

# on retire les nombres
docs = [[token for token in doc if not token.isnumeric()] for doc in docs]
# on retire les token d'une seule lettre
docs = [[token for token in doc if len(token) > 1] for doc in docs]

print(f"Nombre de lemmes : {len(docs)}")
print("MON DOCS")
print(docs)

# Compute bigrams.
from gensim.models import Phrases

# Add bigrams and trigrams to docs (only ones that appear 20 times or more).
bigram = Phrases(docs, min_count=20)
for idx in range(len(docs)):
    for token in bigram[docs[idx]]:
        if '_' in token:
            # Token is a bigram, add to document.
            docs[idx].append(token)

###############################################################################
# We remove rare words and common words based on their *document frequency*.
# Below we remove words that appear in less than 20 documents or in more than
# 50% of the documents. Consider trying to remove words only based on their
# frequency, or maybe combining that with this approach.
#

# Remove rare and common tokens.
from gensim.corpora import Dictionary

# Create a dictionary representation of the documents.
dictionary = Dictionary(docs)

# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=20, no_above=0.5)

###############################################################################
# Finally, we transform the documents to a vectorized form. We simply compute
# the frequency of each word, including the bigrams.
#

# Bag-of-words representation of the documents.
corpus = [dictionary.doc2bow(doc) for doc in docs]

###############################################################################
# Let's see how many tokens and documents we have to train on.
#

print('Number of unique tokens: %d' % len(dictionary))
print('Number of documents: %d' % len(corpus))

###############################################################################
# Training
# --------
#
# We are ready to train the LDA model. We will first discuss how to set some of
# the training parameters.
#
# First of all, the elephant in the room: how many topics do I need? There is
# really no easy answer for this, it will depend on both your data and your
# application. I have used 10 topics here because I wanted to have a few topics
# that I could interpret and "label", and because that turned out to give me
# reasonably good results. You might not need to interpret all your topics, so
# you could use a large number of topics, for example 100.
#
# ``chunksize`` controls how many documents are processed at a time in the
# training algorithm. Increasing chunksize will speed up training, at least as
# long as the chunk of documents easily fit into memory. I've set ``chunksize =
# 2000``, which is more than the amount of documents, so I process all the
# data in one go. Chunksize can however influence the quality of the model, as
# discussed in Hoffman and co-authors [2], but the difference was not
# substantial in this case.
#
# ``passes`` controls how often we train the model on the entire corpus.
# Another word for passes might be "epochs". ``iterations`` is somewhat
# technical, but essentially it controls how often we repeat a particular loop
# over each document. It is important to set the number of "passes" and
# "iterations" high enough.
#
# I suggest the following way to choose iterations and passes. First, enable
# logging (as described in many Gensim tutorials), and set ``eval_every = 1``
# in ``LdaModel``. When training the model look for a line in the log that
# looks something like this::
#
#    2016-06-21 15:40:06,753 - gensim.models.ldamodel - DEBUG - 68/1566 documents converged within 400 iterations
#
# If you set ``passes = 20`` you will see this line 20 times. Make sure that by
# the final passes, most of the documents have converged. So you want to choose
# both passes and iterations to be high enough for this to happen.
#
# We set ``alpha = 'auto'`` and ``eta = 'auto'``. Again this is somewhat
# technical, but essentially we are automatically learning two parameters in
# the model that we usually would have to specify explicitly.
#
###############################################################################
# VISUALISATION
###############################################################################

# Train LDA model.
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Set training parameters.
num_topics = 5
chunksize = 2000
passes = 20
iterations = 400
eval_every = None  # Don't evaluate model perplexity, takes too much time.

# Make an index to word dictionary.
temp = dictionary[0]  # This is only to "load" the dictionary.
id2word = dictionary.id2token

model = LdaModel(
    corpus=corpus,
    id2word=id2word,
    chunksize=chunksize,
    alpha='auto',
    eta='auto',
    iterations=iterations,
    num_topics=num_topics,
    passes=passes,
    eval_every=eval_every
)

top_topics = model.top_topics(corpus)

# Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
print('Average topic coherence: %.4f.' % avg_topic_coherence)

from pprint import pprint
pprint(top_topics)

# Create visualization
lda_visualization = gensimvis.prepare(model, corpus, dictionary)

# Save visualization to HTML file
pyLDAvis.save_html(lda_visualization, 'lda_visualization.html')

# Display visualization
pyLDAvis.display(lda_visualization)