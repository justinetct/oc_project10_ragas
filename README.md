# Évaluez les performances d'un LLM

Assistant d'analyse NBA basé sur une approche RAG (*Retrieval-Augmented Generation*).

L'application permet d'interroger des sources documentaires NBA mixtes : archives Reddit extraites par OCR, documents PDF et fichier Excel de statistiques. Les documents sont indexés dans FAISS, puis interrogés via une interface Streamlit et un modèle Mistral.

## Fonctionnalités

- chargement de documents depuis `inputs/` ;
- extraction OCR des PDF Reddit ;
- lecture d'un fichier Excel de statistiques NBA ;
- génération d'embeddings avec Mistral ;
- création d'un index FAISS local ;
- recherche vectorielle dans les documents ;
- génération de réponses avec Mistral ;
- interface utilisateur Streamlit.

## Structure du dépôt

```text
.
├── MistralChat.py              # Application Streamlit
├── indexer.py                  # Script d'indexation des documents
├── requirements.txt            # Dépendances Python
├── .env.example                # Exemple de configuration sans clé réelle
├── inputs/                     # Documents sources
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
├── utils/
│   ├── config.py               # Configuration des chemins et variables d'environnement
│   ├── data_loader.py          # Chargement OCR / Excel / documents
│   └── vector_store.py         # Création et interrogation de l'index FAISS
├── docs/
│   └── audit_initial.md        # Synthèse de l'audit initial
└── notebooks/
    └── audit.ipynb             # Notebook d'audit initial
```

Le dossier `vector_db/` est généré localement par `python indexer.py`. Il n'est pas versionné car il peut être reconstruit à partir des fichiers présents dans `inputs/`.

## Prérequis

- Python 3.9 ou supérieur ;
- une clé API Mistral ;
- les dépendances listées dans `requirements.txt`.

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

Créer un fichier `.env` à partir du modèle :

```bash
cp .env.example .env
```

Renseigner la clé Mistral dans `.env` :

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

## Indexation des documents

Lancer l'indexation :

```bash
python indexer.py
```

Cette commande lit les documents du dossier `inputs/`, extrait leur contenu, génère les embeddings Mistral et construit l'index FAISS local dans `vector_db/`.

Lors du dernier audit, l'indexation a produit **302 chunks**.

## Lancement de l'application

```bash
streamlit run MistralChat.py
```

L'application est ensuite accessible sur :

```text
http://localhost:8501
```

## Audit initial

L'audit initial est disponible dans :

- `notebooks/audit.ipynb` ;
- `docs/audit_initial.md`.

Il vérifie le fonctionnement des principaux composants : dépendances, données, index FAISS, API Mistral, recherche vectorielle et pipeline RAG complet.

L'audit montre que l'application fonctionne techniquement, mais que les questions chiffrées restent une limite importante. Le système récupère des chunks proches dans l'index FAISS, puis le modèle reformule une réponse sans calcul structuré sur les données Excel.

Exemple observé : le modèle peut répondre **Shai Gilgeous-Alexander — 37,5 %** à la question du meilleur pourcentage à 3 points, alors qu'un extrait contient déjà **Nikola Jokić — 41,7 %**. Cela montre que la recherche vectorielle seule ne calcule pas réellement le maximum d'une colonne.

## Limites connues

- Les PDF Reddit sont extraits par OCR, ce qui peut introduire du bruit dans le texte.
- Le fichier Excel est indexé comme du texte brut, ce qui limite les calculs statistiques fiables.
- Les réponses peuvent être plausibles mais insuffisamment ancrées dans les sources.
- Les questions numériques nécessitent un traitement structuré complémentaire.

## Commandes utiles

```bash
# Activer l'environnement
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Reconstruire l'index FAISS
python indexer.py

# Lancer l'application
streamlit run MistralChat.py
```
