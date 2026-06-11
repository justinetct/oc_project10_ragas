# Évaluez les performances d'un LLM

Assistant d'analyse NBA basé sur une approche RAG (*Retrieval-Augmented Generation*).

L'application permet d'interroger des sources documentaires NBA mixtes : archives Reddit extraites par OCR, documents PDF et fichier Excel de statistiques. Les documents sont indexés dans FAISS, puis interrogés via une interface Streamlit et un modèle Mistral.

> Le rapport de mise en place et d'évaluation est disponible ici : [docs/final_report.md](docs/final_report.md).

Versions repères :

| Nom | Description | Tag Git prévu |
|---|---|---|
| **RAG v1 — baseline** | pipeline RAG initial | `rag-v1-baseline` |
| **RAG v2 — contrôlé** | ajout de Pydantic, Pydantic AI et Logfire | `rag-v2-controlled` |
| **RAG v3 — hybride SQL** | ajout de SQLite et du SQL Tool pour les questions chiffrées | `rag-v3-sql-hybrid` |

## Sommaire

- [Structure du dépôt](#structure-du-dépôt)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Qualité de code](#qualité-de-code)
- [Commandes utiles](#commandes-utiles)
- [Utilisation](#utilisation)
  - [Indexation des documents](#indexation-des-documents)
  - [Lancement de l'application](#lancement-de-lapplication)
- [Évaluation et résultats](#évaluation-et-résultats)
  - [Audit et limites](#audit-et-limites)
  - [Dataset d'évaluation](#dataset-dévaluation)
  - [Validation et génération structurée](#validation-et-génération-structurée)
  - [Baseline RAGAS et versions repères](#baseline-ragas-et-versions-repères)
- [Observabilité](#observabilité)
- [SQL Tool LangChain (lecture seule)](#sql-tool-langchain-lecture-seule)

## Structure du dépôt

```text
.
├── docs/
│   ├── audit_initial.md         # Synthèse de l'audit initial
│   ├── final_report.md          # Rapport de mise en place et d'évaluation
│   └── img/                     # Captures utilisées dans le rapport
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
├── scripts/
│   ├── evaluate_ragas.py        # Baseline RAGAS du prototype sur le dataset
│   └── load_excel_to_db.py      # Construit la base SQLite NBA depuis l'Excel
├── tests/                       # Tests qualité et validation
├── utils/
│   ├── config.py                # Configuration des chemins et variables d'environnement
│   ├── data_loader.py           # Chargement OCR / Excel / documents
│   ├── observability.py         # Configuration optionnelle de Logfire
│   ├── rag_agent.py             # Agent Pydantic AI : génération de la réponse à sortie typée
│   ├── schemas.py               # Modèles Pydantic (validation du pipeline RAG)
│   ├── sql/                     # Base SQLite NBA : modèles, chargement Excel, requêtes
│   └── vector_store.py          # Création et interrogation de l'index FAISS
├── .env.example                 # Exemple de configuration sans clé réelle
├── .gitignore                   # Fichiers locaux exclus du versionnement
├── indexer.py                   # Script d'indexation des documents
├── MistralChat.py               # Application Streamlit
├── poetry.lock                  # Versions verrouillées (reproductibilité)
├── pyproject.toml               # Dépendances et configuration Poetry
└── README.md                    # Documentation principale
```

Le dossier `vector_db/` est généré localement par `python indexer.py`. Il n'est pas versionné car il peut être reconstruit à partir des fichiers présents dans `inputs/`.

---

## Prérequis

- Python 3.11 ou supérieur ;
- Poetry pour la gestion des dépendances ;
- Une clé API Mistral.

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

## Qualité de code

Trois contrôles simples permettent d'éviter de casser le prototype entre deux itérations :

```bash
# Compilation : vérifie la syntaxe de tous les modules
poetry run python -m compileall MistralChat.py indexer.py scripts utils notebooks tests

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
poetry run python scripts/evaluate_ragas.py
```

## Utilisation

### Indexation des documents

Lancer l'indexation :

```bash
poetry run python indexer.py
```

Cette commande lit les documents du dossier `inputs/`, extrait leur contenu, génère les embeddings Mistral et construit l'index FAISS local dans `vector_db/`.

Lors du dernier audit, l'indexation a produit **302 chunks**.

### Lancement de l'application

```bash
poetry run streamlit run MistralChat.py
```

L'application est ensuite accessible sur :

```text
http://localhost:8501
```

## Évaluation et résultats

### Audit et limites

L'audit initial est disponible dans :

- `notebooks/audit.ipynb` ;
- `docs/audit_initial.md`.

Il vérifie les composants principaux : données, index FAISS, API Mistral, recherche vectorielle et pipeline RAG complet.

Le prototype fonctionne, mais les questions chiffrées restent fragiles. FAISS retrouve des passages proches, mais ne calcule pas une statistique dans le fichier Excel.

Exemple observé : le modèle peut répondre **Shai Gilgeous-Alexander — 37,5 %** à la question du meilleur pourcentage à 3 points, alors qu'un extrait contient déjà **Nikola Jokić — 41,7 %**.

Les limites principales sont résumées ici : OCR parfois bruité, Excel indexé comme texte, réponses parfois trop peu ancrées. Le détail est présenté dans le [rapport](docs/final_report.md#audit-initial-et-limites-observées).

### Dataset d'évaluation

Le fichier `evaluation/evaluation_questions.csv` contient le jeu de questions utilisé pour évaluer l'assistant RAG.

Il couvre plusieurs cas : questions simples, complexes, chiffrées, mixtes, bruitées et hors sujet. Il sert de base stable pour comparer les versions du pipeline.

*Une fois la baseline calculée, ce fichier ne doit plus être modifié.*

La méthodologie d'évaluation est détaillée dans le [rapport](docs/final_report.md#3-méthodologie-dévaluation).

### Validation et génération structurée

Le pipeline RAG utilise **Pydantic** et **Pydantic AI** pour valider les données et structurer la réponse du LLM.

Les modèles Pydantic sont définis dans `utils/schemas.py`. Ils valident les documents, les chunks, les contextes récupérés et les réponses.

La génération finale est centralisée dans `utils/rag_agent.py`. L'agent Pydantic AI reçoit la question et les contextes FAISS, puis renvoie une sortie typée `RagAnswerOutput`.

Le même agent est utilisé par `MistralChat.py` et `scripts/evaluate_ragas.py`. L'évaluation mesure donc le même chemin de génération que l'application.

Le fonctionnement et l'impact de cette modification sont expliqués dans le [rapport](docs/final_report.md#5-passage-à-rag-v2--contrôlé).

### Baseline RAGAS et versions repères

Le script `scripts/evaluate_ragas.py` évalue l'assistant RAG sur le jeu de questions figé. Les résultats servent à comparer **RAG v1 — baseline**, **RAG v2 — contrôlé**, puis **RAG v3 — hybride SQL**.

```bash
poetry run python scripts/evaluate_ragas.py
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

La `faithfulness` reste limitée : les réponses ne sont pas toujours assez ancrées dans les sources. Les scores varient d'un run à l'autre, car le juge RAGAS est aussi un LLM. Les valeurs exactes et le détail par catégorie sont disponibles dans `ragas_baseline_summary.json`.

### Robustesse RAG v1 / RAG v2

Pour tenir compte de la variabilité du juge LLM, une expérience A/B a été menée sur 5 runs avec **RAG v1 — baseline** et 5 runs avec **RAG v2 — contrôlé**.

L'expérience montre une hausse de la `faithfulness` moyenne avec Pydantic AI, avec une baisse de l'`answer_relevancy`. Les métriques de contexte restent proches, ce qui indique que l'écart vient surtout de la génération.

Le détail complet est documenté dans le [rapport](docs/final_report.md#robustesse-des-résultats).


## Observabilité

Logfire trace quelques étapes clés du pipeline : recherche vectorielle, génération de réponse RAG et calcul RAGAS.

L'observabilité est **optionnelle** : sans `LOGFIRE_TOKEN`, l'application fonctionne normalement.

Pour l'activer, ajouter les variables suivantes dans `.env` :

```env
LOGFIRE_TOKEN=your_logfire_token_here
LOGFIRE_ENVIRONMENT=local
```

Les variables sont définies dans `.env.example` sans vraie valeur. Aucun secret ne doit être commité.

Le rôle de Logfire dans le pipeline est détaillé dans le [rapport](docs/final_report.md#logfire).

## SQL Tool LangChain (lecture seule)

Pour préparer **RAG v3 — hybride SQL**, le projet fournit un SQL Tool LangChain en lecture seule : `nba_sql_query` (`utils/sql/sql_tool.py`). Il sert aux questions chiffrées : classement, maximum, statistiques d'un joueur.

- Il interroge la base SQLite locale `data/nba.sqlite`, générée par `poetry run python scripts/load_excel_to_db.py`.
- Il n'accepte que des requêtes `SELECT` : mots-clés d'écriture refusés, une seule requête à la fois, nombre de lignes plafonné, connexion ouverte en lecture seule.
- Il complète le RAG texte sans le remplacer : FAISS reste utilisé pour les documents (Reddit/PDF). Le routage automatique entre RAG et SQL sera fait dans une étape séparée.

Le détail (règles de sécurité, exemples de requêtes, limites) est dans le [rapport](docs/final_report.md#7-préparation-de-rag-v3--hybride-sql) et dans `docs/sqlite_schema.md`.
