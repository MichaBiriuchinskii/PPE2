import pickle
import pprint

def lire_fichier_pickle(fichier):
    with open(fichier, 'rb') as file:
        obj = pickle.load(file)
    return obj

# Lire le fichier pickle
obj = lire_fichier_pickle('/Users/mak/PycharmProjects/PPE2/output.pickle')

# Afficher l'objet de manière lisible
#pprint.pprint(obj)




import json

def lire_fichier_json(nom_fichier):
    """
    Lit le contenu d'un fichier JSON et le retourne sous forme de dictionnaire ou de liste.

    Args:
        nom_fichier (str): Le nom du fichier JSON à lire.

    Returns:
        dict or list: Le contenu du fichier JSON sous forme de dictionnaire ou de liste.
    """
    with open(nom_fichier, 'r') as f:
        contenu = json.load(f)
    return contenu

print(lire_fichier_json('/Users/mak/PycharmProjects/PPE2/output.json'))
