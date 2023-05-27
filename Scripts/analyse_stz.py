import stanza
from datastructures import Token, Article

def create_parser():
    stanza.download("fr")  # Télécharger les modèles de la langue française
    return stanza.Pipeline("fr", processors="tokenize,lemma,pos")  # Initialiser le pipeline pour le français

def analyse_article(parser, article: Article) -> Article:
    result = parser((article.titre or "") + "\n" + (article.description or ""))
    output = []
    for sent in result.sentences:
        for word in sent.words:
            output.append(Token(word.text, word.lemma, word.upos))
    article.analyse = output
    return article