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
├── pyproject.toml              # Dépendances et configuration Poetry
├── poetry.lock                 # Versions verrouillées (reproductibilité)
├── requirements.txt            # Ancien fichier pip, conservé temporairement pendant la migration
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

- Python 3.11 ou supérieur ;
- [Poetry](https://python-poetry.org/) pour la gestion des dépendances ;
- une clé API Mistral.

## Installation

Installer les dépendances avec Poetry (crée automatiquement un environnement virtuel) :

```bash
poetry install
```

Créer un fichier `.env` à partir du modèle :

```bash
cp .env.example .env
```

Renseigner la clé Mistral dans `.env` :

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

> Note : l'ancien `requirements.txt` est conservé temporairement pendant la migration vers Poetry. La référence des dépendances est désormais `pyproject.toml` / `poetry.lock`.

## Indexation des documents

Lancer l'indexation :

```bash
poetry run python indexer.py
```

Cette commande lit les documents du dossier `inputs/`, extrait leur contenu, génère les embeddings Mistral et construit l'index FAISS local dans `vector_db/`.

Lors du dernier audit, l'indexation a produit **302 chunks**.

## Lancement de l'application

```bash
poetry run streamlit run MistralChat.py
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

## Qualité de code

Trois contrôles simples permettent d'éviter de casser le prototype entre deux itérations :

```bash
# Compilation : vérifie la syntaxe de tous les modules
poetry run python -m compileall MistralChat.py indexer.py utils notebooks tests

# Linter
poetry run ruff check .

# Tests de non-régression (sans appel API ni OCR)
poetry run pytest
```

Les tests du dossier `tests/` sont volontairement légers : ils vérifient la configuration, la présence des fichiers d'entrée et l'import des modules sans effet de bord. Ils ne déclenchent **aucun** appel à l'API Mistral, ni l'OCR, ni la reconstruction de l'index FAISS.

## Commandes utiles

```bash
# Installer les dépendances
poetry install

# Reconstruire l'index FAISS
poetry run python indexer.py

# Lancer l'application
poetry run streamlit run MistralChat.py
```

Voir aussi la section **Qualité de code** ci-dessus (`compileall`, `ruff`, `pytest`).
