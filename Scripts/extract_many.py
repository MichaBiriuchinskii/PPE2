import argparse
import re
import sys
from datetime import date
from pathlib import Path
from typing import Optional, List

from export_json import write_json
from tqdm import tqdm

# import analyse_tk as analyse
# import analyse_stz as analyse
import analyse_sp as analyse
import extract_un_fil as euf
from datastructures import Corpus
from export_pickle import write_pickle
from export_xml import write_xml

MONTHS = ["Jan",
          "Feb",
          "Mar",
          "Apr",
          "May",
          "Jun",
          "Jul",
          "Aug",
          "Sep",
          "Oct",
          "Nov",
          "Dec"]

DAYS = [f"{x:02}" for x in range(1,32)]

CAT_CODES =  {
            'une' : '0,2-3208,1-0,0',
            'international' : '0,2-3210,1-0,0',
            'europe' : '0,2-3214,1-0,0',
            'societe' :	'0,2-3224,1-0,0',
            'idees'	: '0,2-3232,1-0,0',
            'economie':	'0,2-3234,1-0,0',
            'actualite-medias':	'0,2-3236,1-0,0',
            'sport': '0,2-3242,1-0,0',
            'planete': '0,2-3244,1-0,0',
            'culture': '0,2-3246,1-0,0',
            'livres' : '0,2-3260,1-0,0',
            'cinema' : '0,2-3476,1-0,0',
            'voyage' : '0,2-3546,1-0,0',
            'technologies': '0,2-651865,1-0,0',
            'politique' : '0,57-0,64-823353,0',
            'sciences' : 'env_sciences'
            }


def categorie_of_ficname(ficname: str)-> Optional[str]:
    for nom, code in CAT_CODES.items():
        if code in ficname:
            return nom
    return None

def convert_month(m:str) -> int:
   return MONTHS.index(m) + 1

def parcours_path(corpus_dir:Path, categories: Optional[List[str]]=None, start_date: Optional[date]=None, end_date: Optional[date]=None):
    if categories is not None and len(categories) > 0:
        categories = [CAT_CODES[c] for c in categories]
    else:
        categories = CAT_CODES.values() # on prend tout

    for month_dir in corpus_dir.iterdir():
        if month_dir.name not in MONTHS:
            # on ignore les dossiers qui ne sont pas des mois
            continue
        m = convert_month(month_dir.name)
        for day_dir in month_dir.iterdir():
            if day_dir.name not in DAYS:
                # on ignore les dossiers qui ne sont pas des jours
                continue
            d = date.fromisoformat(f"2022-{m:02}-{day_dir.name}")
            if (start_date is None or start_date <= d) and (end_date is None or end_date >= d):
                for time_dir in day_dir.iterdir():
                    if re.match(r"\d\d-\d\d-\d\d", time_dir.name):
                        for fic in time_dir.iterdir():
                            if fic.name.endswith(".xml") and any([c in fic.name for c in categories]):
                                c = categorie_of_ficname(fic.name)
                                yield fic, d.isoformat(), c




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", help="méthode de parsing (et, re ou fp)", default="et")
    parser.add_argument("-s", help="start date (iso format)", default="2022-01-01")
    parser.add_argument("-e", help="end date (iso format)", default="2023-01-01")
    parser.add_argument("-o", help="output file", required=True)
    parser.add_argument("-f", help="output format (xml by default)", default="xml")
    parser.add_argument("corpus_dir", help="root dir of the corpus data")
    parser.add_argument("categories",nargs="*", help="catégories à retenir")
    args = parser.parse_args()
    if args.m == 'et':
        func = euf.extract_et
    elif args.m == 're':
        func = euf.extract_re
    elif args.m == 'fp':
        func = euf.extract_feedparser
    else:
        print("méthode non disponible", file=sys.stderr)
        sys.exit()
    # création du corpus
    corpus = Corpus(args.categories, args.s, args.e, args.corpus_dir, [])
    for f, d, c in parcours_path(Path(args.corpus_dir),
            start_date=date.fromisoformat(args.s),
            end_date=date.fromisoformat(args.e),
            categories=args.categories):
        # fichier par fichier
        for article in func(f, d, c):
            # parcours article par article
            corpus.articles.append(article)
    parser = analyse.create_parser()
    for a in tqdm(corpus.articles):
        analyse.analyse_article(parser, a)

    # -> corpus
    if args.f == "xml":
        write_xml(corpus, args.o)
    elif args.f == "json":
        write_json(corpus, args.o)
    elif args.f == "pickle":
        write_pickle(corpus, args.o)

    else:
        # note: ce serait mieux de tester au début de l'execution
        print("format de sortie non reconnu", file=sys.stderr)
