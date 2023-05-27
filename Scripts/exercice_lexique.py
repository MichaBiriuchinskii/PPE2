#!/usr/bin/python3

import os
import re
import sys
import argparse

def liste_chaines(liste_files: list) -> list:
    """
        Construit une liste de chaiînes de caractères, où chaque
        chaîne correspond au contenu d'un fichier
    """
    liste = list()


    for file in liste_files:
        with open(file, 'r', encoding="utf8", errors='ignore') as file:
            text = file.read()
            if  re.search("\x00.*", text): # nettoyage
                continue
            else:
                liste.append(text)

    return liste

def compter_occurrences(liste_chaines: list) -> dict:
    """
        Compte le nombre d'occurrences de chaque mot dans une liste de chaînes
        et renvoie un dictionnaire associant chaque mot à son nombre d'occurrences.
    """
    occurrences = {}
    for chaine in liste_chaines:
        mots = chaine.split()
        for mot in mots:
            if mot in occurrences:
                occurrences[mot] += 1
            else:
                occurrences[mot] = 1
    return occurrences


#Partie r3

def doc_freq(corpus: list) -> dict:
    resultat = {}
    for doc in corpus:
        words = set(doc.split())
        for word in words:
            if word in resultat:
                resultat[word] += 1
            else:
                resultat[word] = 1
    return resultat

#Partie r2 ex2 Micha

def main():
    try:
        parser_r1 = argparse.ArgumentParser(description = "permettre la lecture du corpus comme une liste de fichiers en arguments")
        parser_r1.add_argument('pathname')
        args = parser_r1.parse_args()

    except:
        # r3
        avant_pipe = sys.stdin.readlines() # readlines récupère la sortie de ls
        if re.search('\.txt', avant_pipe[0]):
            for i,j in enumerate(avant_pipe):
                avant_pipe[i]=j.replace("\n","")
            corpus = liste_chaines(avant_pipe)
        else:
            # r2
            corpus = [line.rstrip('\n') for line in sys.stdin]
    else:
        (path, extension) = args.pathname.split('*')
        fichiers = []
        for file in os.listdir(path):
            if re.search('.*\.txt$', file):
                fichiers.append(path+file)
        corpus = liste_chaines(fichiers)

    print("doc freq")
    for k, v in doc_freq(corpus).items():
        print(f"{k}: {v}")
    print("term freq")
    occurrences = compter_occurrences(corpus)
    for k, v in occurrences.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
