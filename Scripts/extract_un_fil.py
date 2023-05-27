#!/usr/bin/python3

import sys
import re
from pathlib import Path
import feedparser
from xml.etree import ElementTree as et
from datastructures import Article


regex_item = re.compile("<item><title>(.*?)<\/title>.*?<description>(.*?)<\/description>")


def nettoyage(texte):
    texte_net = re.sub("<!\[CDATA\[(.*?)\]\]>", "\\1", texte) 
    return texte_net

def extract_re(fichier_rss, date, categorie):
    with open(fichier_rss, "r") as input_rss:
        lignes = input_rss.readlines()
        texte = "".join(lignes)
        for m in re.finditer(regex_item, texte):
            titre_net = nettoyage(m.group(1))
            description_net = nettoyage(m.group(2))
            yield Article(titre_net, description_net, date, categorie, [])
    

def extract_feedparser(fichier_rss, date, categorie):
    feed = feedparser.parse(fichier_rss)
    for entry in feed['entries']:
        yield Article(entry['title'], entry['summary'], date, categorie, [])

def extract_et(fichier_rss, date, categorie):
    text = Path(fichier_rss).read_text()
    if len(text) > 0:
        xml = et.parse(fichier_rss)
        root = xml.getroot()
        for item in root.findall(".//item"):
            title = item.find("title").text
            desc = item.find("description").text
            yield Article(title, desc, date, categorie, [])

def main(fichier_rss, extract_fun):
    for i, (titre, description) in enumerate(extract_fun(fichier_rss, "", "")):
        print(i,titre)
        print(description)
        print("---")

if __name__ == "__main__":
    fichier_rss = sys.argv[1]
    print("démo RE")
    main(fichier_rss, extract_re)
    print("démo feedparser")
    main(fichier_rss, extract_feedparser)
    print("démo etree")
    main(fichier_rss, extract_et)

