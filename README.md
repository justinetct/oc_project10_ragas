# Évaluez les performances d'un LLM

Assistant d'analyse NBA basé sur une approche RAG (*Retrieval-Augmented Generation*).

L'application permet d'interroger des sources documentaires NBA mixtes : archives Reddit extraites par OCR, documents PDF et fichier Excel de statistiques. Les documents sont indexés dans FAISS, puis interrogés via une interface Streamlit et un modèle Mistral.

## Sommaire

- [Structure du dépôt](#structure-du-dépôt)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Qualité de code](#qualité-de-code)
- [Commandes utiles](#commandes-utiles)
- [Indexation des documents](#indexation-des-documents)
- [Lancement de l'application](#lancement-de-lapplication)
- [Audit initial](#audit-initial)
- [Dataset d'évaluation](#dataset-dévaluation)
- [Validation et génération structurée (Pydantic)](#validation-et-génération-structurée-pydantic)
- [Baseline RAGAS](#baseline-ragas)
- [Observabilité (Logfire)](#observabilité-logfire)

## Structure du dépôt

```text
.
├── docs/
│   └── audit_initial.md         # Synthèse de l'audit initial
├── evaluation/
│   ├── evaluation_questions.csv # Dataset d'évaluation versionné
│   └── results/                 # Résultats de la baseline RAGAS
├── inputs/                      # Documents sources
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
├── notebooks/
│   ├── audit.ipynb              # Notebook d'audit initial
│   └── ragas_baseline_results.ipynb  # Illustration de la baseline RAGAS
├── tests/                       # Tests qualité et validation
├── utils/
│   ├── config.py                # Configuration des chemins et variables d'environnement
│   ├── data_loader.py           # Chargement OCR / Excel / documents
│   ├── observability.py         # Configuration optionnelle de Logfire
│   ├── rag_agent.py             # Agent Pydantic AI : génération de la réponse à sortie typée
│   ├── schemas.py               # Modèles Pydantic (validation du pipeline RAG)
│   └── vector_store.py          # Création et interrogation de l'index FAISS
├── .env.example                 # Exemple de configuration sans clé réelle
├── .gitignore                   # Fichiers locaux exclus du versionnement
├── evaluate_ragas.py            # Baseline RAGAS du prototype sur le dataset
├── indexer.py                   # Script d'indexation des documents
├── MistralChat.py               # Application Streamlit
├── poetry.lock                  # Versions verrouillées (reproductibilité)
├── pyproject.toml               # Dépendances et configuration Poetry
├── README.md                    # Documentation principale
└── requirements.txt             # Ancien fichier pip, conservé temporairement pendant la migration
```

Le dossier `vector_db/` est généré localement par `python indexer.py`. Il n'est pas versionné car il peut être reconstruit à partir des fichiers présents dans `inputs/`.

---
## Prérequis

- Python 3.11 ou supérieur ;
- [Poetry](https://python-poetry.org/) pour la gestion des dépendances ;
- une clé API Mistral.

## Installation

Installer les dépendances avec Poetry :

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

## Qualité de code

Trois contrôles simples permettent d'éviter de casser le prototype entre deux itérations :

```bash
# Compilation : vérifie la syntaxe de tous les modules
poetry run python -m compileall MistralChat.py indexer.py utils notebooks tests evaluate_ragas.py

# Linter
poetry run ruff check .

# Tests qualité et validation (sans appel API ni OCR)
poetry run pytest
```

Les tests du dossier `tests/` sont légers : ils vérifient la configuration, la présence des fichiers d'entrée, la structure du dataset d'évaluation, l'import des modules sans effet de bord et quelques comportements de non-régression du vector store. Ils ne déclenchent **aucun** appel à l'API Mistral, ni l'OCR, ni la reconstruction de l'index FAISS.

Tests actuellement présents :

- `test_config.py` — configuration et chemins principaux ;
- `test_inputs.py` — présence des documents sources ;
- `test_imports.py` — imports sans appel API ni OCR ;
- `test_vector_store.py` — comportements clés du vector store ;
- `test_schemas.py` — validation Pydantic des objets du pipeline RAG ;
- `test_observability.py` — configuration Logfire optionnelle ;
- `test_evaluation_dataset.py` — structure du dataset d'évaluation ;
- `test_rag_agent.py` — agent Pydantic AI : import, construction du contexte, sortie typée.

## Commandes utiles

```bash
# Installer les dépendances
poetry install

# Reconstruire l'index FAISS
poetry run python indexer.py

# Lancer l'application
poetry run streamlit run MistralChat.py

# Lancer la baseline RAGAS
poetry run python evaluate_ragas.py
```

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

---
## Audit initial

L'audit initial est disponible dans :

- `notebooks/audit.ipynb` ;
- `docs/audit_initial.md`.

Il vérifie le fonctionnement des principaux composants : dépendances, données, index FAISS, API Mistral, recherche vectorielle et pipeline RAG complet.

L'audit montre que l'application fonctionne techniquement, mais que les questions chiffrées restent une limite importante. Le système récupère des chunks proches dans l'index FAISS, puis le modèle reformule une réponse sans calcul structuré sur les données Excel.

Exemple observé : le modèle peut répondre **Shai Gilgeous-Alexander — 37,5 %** à la question du meilleur pourcentage à 3 points, alors qu'un extrait contient déjà **Nikola Jokić — 41,7 %**. Cela montre que la recherche vectorielle seule ne calcule pas réellement le maximum d'une colonne.

### Limites

- Les PDF Reddit sont extraits par OCR, ce qui peut introduire du bruit dans le texte.
- Le fichier Excel est indexé comme du texte brut, ce qui limite les calculs statistiques fiables.
- Les réponses peuvent être plausibles mais insuffisamment ancrées dans les sources.
- Les questions numériques nécessitent un traitement structuré complémentaire.

---
## Dataset d'évaluation

Le fichier `evaluation/evaluation_questions.csv` contient le jeu de questions utilisé pour évaluer l'assistant RAG.

Chaque ligne correspond à une question, avec sa catégorie, le comportement attendu, une réponse de référence courte, une indication de source et un champ `requires_sql_future`.

Le dataset couvre plusieurs cas : questions simples, complexes, chiffrées, mixtes, bruitées et hors sujet. Il sert de base stable pour comparer la baseline RAGAS avec la future version améliorée par SQL.

*Une fois la baseline calculée, ce fichier ne doit plus être modifié.*

---
## Validation et génération structurée (Pydantic)

Le pipeline RAG utilise **Pydantic** et **Pydantic AI** pour sécuriser les données manipulées par le prototype et structurer la réponse produite par le LLM.

### Pydantic : validation des objets du pipeline

Les modèles définis dans `utils/schemas.py` décrivent les principaux objets qui circulent dans le pipeline RAG :

- documents chargés depuis les sources ;
- chunks indexés dans FAISS ;
- contextes récupérés avec leur score ;
- réponse finale associée à la question et aux contextes utilisés.

Cette validation permet de détecter plus tôt les incohérences de structure, les champs manquants ou les réponses vides.

### Pydantic AI : génération à sortie typée

La génération de la réponse finale est centralisée dans `utils/rag_agent.py` avec un **agent Pydantic AI** basé sur Mistral.

L'agent reçoit :

- la question utilisateur ;
- les contextes récupérés par FAISS ;
- un prompt RAG commun à l'application et à l'évaluation.

Il renvoie une sortie typée `RagAnswerOutput`, validée par Pydantic, avec une réponse non vide. Cette sortie est ensuite compatible avec le modèle `RagAnswer` utilisé dans le reste du pipeline.

Le même agent est utilisé dans :

- `MistralChat.py` pour l'application Streamlit ;
- `evaluate_ragas.py` pour la baseline RAGAS.

Ainsi, l'évaluation mesure le même chemin de génération que celui utilisé par l'application.

---
## Baseline RAGAS

Le script `evaluate_ragas.py` évalue l'assistant RAG sur le dataset figé.

```bash
poetry run python evaluate_ragas.py
```

Prérequis : un fichier `.env` avec `MISTRAL_API_KEY` et un index FAISS déjà construit (`poetry run python indexer.py`).

Les métriques utilisées sont :

- `faithfulness` — la réponse est-elle ancrée dans les contextes récupérés ?
- `answer_relevancy` — la réponse répond-elle réellement à la question ?
- `context_precision` — les contextes récupérés sont-ils pertinents au regard de la référence ?
- `context_recall` — la référence est-elle couverte par les contextes récupérés ?


Les résultats sont écrits dans `evaluation/results/` :

- `ragas_baseline_results.csv` — résultats détaillés par question ;
- `ragas_baseline_summary.json` — synthèse globale et par catégorie.

Le notebook `notebooks/ragas_baseline_results.ipynb` permet de visualiser ces résultats sans relancer l'évaluation.

Scores moyens de la baseline actuelle (juge RAGAS `mistral-large-latest`, 15 questions) :

| Métrique | Score moyen |
|---|---:|
| `faithfulness` | 0,35 |
| `answer_relevancy` | 0,55 |
| `context_precision` | 0,36 |
| `context_recall` | 0,43 |

La faible `faithfulness` confirme que les réponses restent insuffisamment ancrées dans les sources. Ces scores varient de ~0,04 à 0,08 d'un run à l'autre (juge LLM non déterministe) : on ne les sur-interprète donc pas à la 2ᵉ décimale. Valeurs exactes et détail par catégorie dans `ragas_baseline_summary.json`.

### Robustesse de la baseline

Pour tenir compte de la variabilité du juge LLM, une expérience A/B a été menée sur 5 runs avec l'agent Pydantic AI et 5 runs avec l'ancien pipeline de génération directe.

L'expérience montre que l'agent Pydantic AI améliore la `faithfulness` moyenne, mais réduit l'`answer_relevancy`. Les métriques de contexte restent stables, ce qui confirme que la différence vient de la génération et non de la récupération FAISS.

Le détail complet de l'expérience est documenté dans le rapport final.
---
## Observabilité (Logfire)

Logfire est utilisé pour tracer quelques étapes clés du pipeline : recherche vectorielle, génération de réponse RAG et calcul RAGAS.

L'observabilité est **optionnelle** : sans `LOGFIRE_TOKEN`, l'application fonctionne normalement en mode local silencieux.

Pour l'activer, ajouter les variables suivantes dans `.env` :

```env
LOGFIRE_TOKEN=your_logfire_token_here
LOGFIRE_ENVIRONMENT=local
```

Les variables sont définies dans `.env.example` sans vraie valeur. Aucun secret ne doit être commité.