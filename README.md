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
- [Routage RAG / SQL](#routage-rag--sql)
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
│   ├── routing.md               # Routes RAG / SQL / hybride (référence)
│   ├── sqlite_schema.md         # Schéma de la base SQLite NBA
│   └── img/                     # Captures utilisées dans le rapport
├── evaluation/
│   ├── evaluation_questions.csv # Jeu figé E01-E15 (comparaison officielle)
│   └── results/                 # Résultats RAGAS (par condition)
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
│   ├── evaluate_ragas.py        # Évaluation RAGAS (pipeline routé ou baseline RAG)
│   └── load_excel_to_db.py      # Construit la base SQLite NBA depuis l'Excel
├── tests/                       # Tests qualité et validation
├── utils/
│   ├── config.py                # Configuration des chemins et variables d'environnement
│   ├── data_loader.py           # Chargement OCR / Excel / documents
│   ├── observability.py         # Configuration optionnelle de Logfire
│   ├── rag_agent.py             # Agent Pydantic AI : génération de la réponse à sortie typée
│   ├── router.py                # Routage RAG / SQL / hybride / hors-sujet
│   ├── schemas.py               # Modèles Pydantic (validation du pipeline RAG)
│   ├── sql/                     # Base SQLite NBA : modèles, chargement Excel, requêtes, mapping
│   ├── text.py                  # Normalisation et détection de mots-clés (routage)
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

## Routage RAG / SQL

Depuis **RAG v3 — hybride SQL**, l'assistant choisit automatiquement le chemin selon la question :

- **RAG texte** (FAISS) pour les questions documentaires, opinions et discussions Reddit ;
- **SQL chiffres** (SQL Tool en lecture seule) pour les questions chiffrées : classements, maximum, total, fiche d'un joueur ;
- **Hybride** pour les questions mixtes : le chiffre est récupéré par SQL (fait vérifié), puis la réponse est rédigée par le LLM ;
- **Hors périmètre** : refus poli pour les questions hors NBA.

L'orchestration est dans `utils/router.py`, le mapping question → SQL dans `utils/sql/nba_intents.py`. Le SQL n'est **jamais** écrit par le LLM : ce sont des requêtes prédéfinies, exécutées par le SQL Tool sécurisé. Pour le 3P%, un filtre de volume (≥ 100 tentatives) évite l'artefact d'un joueur à 100 % sur 1 tir. L'interface Streamlit affiche discrètement la route utilisée.

Préalable aux routes SQL / hybride — construire la base SQLite :

```bash
poetry run python scripts/load_excel_to_db.py
```

Le mode hybride a deux variantes, réglées par la variable `HYBRID_MODE` (`sql_only` par défaut, ou `sql_with_rag_context`). Détail des routes, cas couverts et limites : [docs/routing.md](docs/routing.md).

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

Le fichier `evaluation/evaluation_questions.csv` contient le jeu **figé E01–E15**, identique depuis la baseline : c'est le seul jeu de la comparaison officielle v1 → v2 → v3. Il couvre les cas simples, complexes, chiffrés, mixtes, bruités et hors sujet.

*Le jeu figé E01–E15 ne doit plus être modifié une fois la baseline calculée.*

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

Depuis le routage, l'évaluation passe par le **même pipeline que l'application** (`utils/router.py`) et porte sur le jeu figé E01–E15. Trois conditions peuvent être comparées sur le même jeu de questions :

```bash
poetry run python scripts/evaluate_ragas.py --eval-mode baseline_rag
HYBRID_MODE=sql_only poetry run python scripts/evaluate_ragas.py --eval-mode routed
HYBRID_MODE=sql_with_rag_context poetry run python scripts/evaluate_ragas.py --eval-mode routed
```

Chaque condition écrit ses propres fichiers `ragas_<condition>_results.csv` / `_summary.json` (colonnes `route` et `mode` incluses). Le détail et la synthèse des résultats figurent dans le [rapport](docs/final_report.md).

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
- Il complète le RAG texte sans le remplacer : FAISS reste utilisé pour les documents (Reddit/PDF). Le routage automatique entre RAG, SQL et hybride est désormais branché (voir [Routage RAG / SQL](#routage-rag--sql) et [docs/routing.md](docs/routing.md)).

Le détail (règles de sécurité, exemples de requêtes, limites) est dans le [rapport](docs/final_report.md) et dans `docs/sqlite_schema.md`.
