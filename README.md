# Évaluez les performances d'un LLM

Assistant d'analyse NBA basé sur une approche RAG (*Retrieval-Augmented Generation*).

L'application permet d'interroger des sources NBA mixtes : archives Reddit extraites par OCR, documents PDF et fichier Excel de statistiques. Les documents texte sont indexés dans FAISS ; les statistiques Excel sont chargées dans SQLite pour les questions chiffrées. L'ensemble est accessible via une interface Streamlit et un modèle Mistral.

Versions repères :

| Nom | Description | Tag Git prévu |
|---|---|---|
| **RAG v1 — baseline** | pipeline RAG initial | `rag-v1-baseline` |
| **RAG v2 — contrôlé** | ajout de Pydantic, Pydantic AI et Logfire | `rag-v2-controlled` |
| **RAG v3 — hybride SQL** | routage RAG / SQL avec SQLite et SQL Tool pour les questions chiffrées | `rag-v3-sql-hybrid` |

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
- [Mode expérimental : SQL généré par le LLM](#mode-expérimental--sql-généré-par-le-llm)

## Structure du dépôt

```text
.
├── docs/
│   ├── audit_initial.md         # Synthèse de l'audit initial
│   ├── final_report.md          # Rapport de mise en place et d'évaluation
│   └── img/                     # Captures utilisées dans le rapport
├── evaluation/
│   ├── evaluation_questions.csv              # Jeu figé E01-E15 (comparaison officielle)
│   ├── evaluation_questions_sql_extended.csv # Jeu étendu de questions chiffrées (analyse SQL)
│   └── results/                 # Résultats RAGAS (par condition + runs de variance)
├── inputs/                      # Documents sources
│   ├── Reddit 1.pdf
│   ├── Reddit 2.pdf
│   ├── Reddit 3.pdf
│   ├── Reddit 4.pdf
│   └── regular NBA.xlsx
├── notebooks/
│   ├── audit.ipynb                  # Notebook d'audit initial
│   ├── ragas_baseline_results.ipynb # Illustration de la baseline RAGAS
│   └── sql_modes_analysis.ipynb     # Comparaison des modes SQL (contrôlé / hybride / LLM)
├── scripts/
│   ├── evaluate_ragas.py              # Évaluation RAGAS (pipeline routé ou baseline RAG)
│   ├── compare_ragas_runs.py         # Compare les résumés RAGAS de plusieurs conditions
│   ├── aggregate_variance_runs.py    # Moyenne ± écart-type sur plusieurs runs (juge bruité)
│   ├── compare_sql_modes_extended.py # Compare contrôlé vs LLM→SQL sur le jeu étendu (sans RAGAS)
│   ├── analyze_llm_sql_generation.py # Trace les requêtes générées par le LLM (sans RAGAS)
│   ├── run_all_ragas.sh              # Lance tous les runs (variance ×5 + passes extra)
│   └── load_excel_to_db.py           # Construit la base SQLite NBA depuis l'Excel
├── tests/                       # Tests qualité et validation (sans appel API)
├── utils/
│   ├── config.py                # Configuration app / RAG / SQL / Logfire
│   ├── ragas_config.py          # Configuration dédiée à l'évaluation RAGAS
│   ├── results_io.py            # Lecture des résultats d'évaluation (notebook + tests)
│   ├── data_loader.py           # Chargement OCR / Excel / documents
│   ├── observability.py         # Configuration optionnelle de Logfire
│   ├── rag_agent.py             # Agent Pydantic AI : génération de la réponse à sortie typée
│   ├── router.py                # Routage RAG / SQL / hybride / hors-sujet
│   ├── schemas.py               # Modèles Pydantic (validation du pipeline RAG)
│   ├── sql/                     # Base SQLite NBA + SQL Tool + mode LLM→SQL
│   │   ├── nba_intents.py           # Mapping question → SQL contrôlé (liste blanche)
│   │   ├── sql_tool.py              # SQL Tool LangChain en lecture seule
│   │   ├── llm_sql_generator.py     # Mode expérimental : le LLM propose une requête SQL
│   │   └── llm_sql_pipeline.py      # Valide puis exécute la requête générée (lecture seule)
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

Les tests du dossier `tests/` sont légers et ne déclenchent **aucun** appel à l'API Mistral, ni l'OCR, ni la reconstruction de l'index FAISS. Ils couvrent la configuration, le routage RAG / SQL / hybride, la sécurité du SQL Tool en lecture seule, le mode expérimental LLM→SQL, la validation Pydantic et la lecture des résultats RAGAS.

Pour voir la liste à jour et tout exécuter :

```bash
ls tests/            # liste des fichiers de tests
poetry run pytest -q # exécution (sans API)
```

## Commandes utiles

```bash
# Installer les dépendances
poetry install

# Reconstruire l'index FAISS
poetry run python indexer.py

# Lancer l'application
poetry run streamlit run MistralChat.py

# Construire la base SQLite NBA
poetry run python scripts/load_excel_to_db.py

# Lancer l'évaluation RAGAS avec le pipeline routé
poetry run python scripts/evaluate_ragas.py --eval-mode routed
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

Dans **RAG v3 — hybride SQL**, le routeur choisit le traitement selon la question :

- **RAG texte** (FAISS) pour les questions documentaires, opinions et discussions Reddit ;
- **SQL chiffres** (SQL Tool en lecture seule) pour les questions chiffrées : classements, maximum, total, fiche d'un joueur ;
- **Hybride** pour les questions mixtes : le chiffre est récupéré par SQL (fait vérifié), puis la réponse est rédigée par le LLM ;
- **Hors périmètre** : refus poli pour les questions hors NBA.

L'orchestration est dans `utils/router.py`, le mapping question → SQL dans `utils/sql/nba_intents.py`. Par **défaut** (mode `controlled`), le SQL n'est **jamais écrit par le LLM** : les requêtes sont construites depuis des intentions et des colonnes sur liste blanche. Un mode **expérimental** (`SQL_GENERATION_MODE=llm`) laisse le LLM *proposer* la requête, mais **uniquement dans ce mode** — et dans tous les cas, l'exécution passe **toujours** par le SQL Tool sécurisé en **lecture seule** (voir la section dédiée plus bas). Pour le 3P%, un filtre de volume (≥ 100 tentatives) évite l'artefact d'un joueur à 100 % sur 1 tir. L'interface Streamlit affiche discrètement la route utilisée.

Préalable aux routes SQL / hybride — construire la base SQLite :

```bash
poetry run python scripts/load_excel_to_db.py
```

Le mode hybride a deux variantes, réglées par la variable `HYBRID_MODE` (`sql_only` par défaut, ou `sql_with_rag_context`). Le détail des routes, cas couverts et limites est présenté dans le [rapport](docs/final_report.md#7-rag-v3--hybride-sql).

## Évaluation et résultats

### Audit et limites

L'audit initial est disponible dans :

- `notebooks/audit.ipynb` ;
- `docs/audit_initial.md`.

Il vérifie les composants principaux : données, index FAISS, API Mistral, recherche vectorielle et pipeline RAG complet.

Le prototype fonctionne, mais les questions chiffrées restent fragiles. FAISS retrouve des passages proches, mais ne calcule pas une statistique dans le fichier Excel.

Exemple observé : le modèle peut répondre **Shai Gilgeous-Alexander — 37,5 %** à la question du meilleur pourcentage à 3 points, alors qu'un extrait contient déjà **Nikola Jokić — 41,7 %**.

Les limites principales sont les suivantes : OCR parfois bruité, Excel indexé comme texte dans les premières versions, réponses parfois trop peu ancrées.

### Dataset d'évaluation

Le fichier `evaluation/evaluation_questions.csv` contient le jeu **figé E01–E15**, identique depuis la baseline : c'est le seul jeu de la comparaison officielle v1 → v2 → v3. Il couvre les cas simples, complexes, chiffrés, mixtes, bruités et hors sujet.

*Le jeu figé E01–E15 ne doit plus être modifié une fois la baseline calculée.*

Un **jeu étendu séparé**, `evaluation/evaluation_questions_sql_extended.csv` (~47 questions chiffrées), sert uniquement à analyser plus finement les modes SQL. Il **ne remplace pas** le jeu figé et ne sert pas à la comparaison officielle.

La méthodologie d'évaluation repose sur RAGAS, avec un juge LLM et quatre métriques : `faithfulness`, `answer_relevancy`, `context_precision` et `context_recall`.

### Validation et génération structurée

Le pipeline RAG utilise **Pydantic** et **Pydantic AI** pour valider les données et structurer la réponse du LLM.

Les modèles Pydantic sont définis dans `utils/schemas.py`. Ils valident les documents, les chunks, les contextes récupérés et les réponses.

La génération finale est centralisée dans `utils/rag_agent.py`. L'agent Pydantic AI reçoit la question et les contextes FAISS, puis renvoie une sortie typée `RagAnswerOutput`.

Le même agent est utilisé par `MistralChat.py` et `scripts/evaluate_ragas.py`. L'évaluation mesure donc le même chemin de génération que l'application.

### Baseline RAGAS et versions repères

Le script `scripts/evaluate_ragas.py` évalue l'assistant sur le jeu de questions figé. Les résultats servent à comparer **RAG v1 — baseline**, **RAG v2 — contrôlé**, puis **RAG v3 — hybride SQL**.

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

Chaque condition écrit ses propres fichiers `ragas_<condition>_results.csv` / `_summary.json` (colonnes `route` et `mode` incluses).

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

Cette comparaison sert à vérifier que l'évolution du pipeline ne repose pas sur un seul run isolé.

### Comparaison des modes SQL (RAG v3)

Trois conditions sont comparées sur les questions chiffrées : `controlled_sql`, `controlled_hybrid` et `llm_sql`. Comme le juge RAGAS est lui-même un LLM (résultats bruités), chaque condition est lancée **5 fois** : `scripts/run_all_ragas.sh` enchaîne les runs, et `scripts/aggregate_variance_runs.py` calcule la **moyenne ± écart-type**.

Des métriques **supplémentaires optionnelles** (désactivées par défaut) peuvent être activées via `RAGAS_EXTRA_METRICS` :

```bash
# answer_correctness (justesse vs réponse de référence) et aspect_critic (respect des limites des données)
RAGAS_EXTRA_METRICS=answer_correctness,aspect_critic poetry run python scripts/evaluate_ragas.py --eval-mode routed
```

Le notebook `notebooks/sql_modes_analysis.ipynb` lit tous ces résultats (sans relancer l'API) et présente la comparaison par route et par type de question.


## Observabilité

Logfire trace quelques étapes clés du pipeline : recherche vectorielle, génération de réponse RAG et calcul RAGAS.

L'observabilité est **optionnelle** : sans `LOGFIRE_TOKEN`, l'application fonctionne normalement.

Pour l'activer, ajouter les variables suivantes dans `.env` :

```env
LOGFIRE_TOKEN=your_logfire_token_here
LOGFIRE_ENVIRONMENT=local
```

Les variables sont définies dans `.env.example` sans vraie valeur. Aucun secret ne doit être commité.

## SQL Tool LangChain (lecture seule)

Pour **RAG v3 — hybride SQL**, le projet fournit un SQL Tool LangChain en lecture seule : `nba_sql_query` (`utils/sql/sql_tool.py`). Il sert aux questions chiffrées : classement, maximum, minimum, statistiques d'un joueur.

- Il interroge la base SQLite locale `data/nba.sqlite`, générée par `poetry run python scripts/load_excel_to_db.py`.
- Il n'accepte que des requêtes `SELECT` : mots-clés d'écriture refusés, une seule requête à la fois, nombre de lignes plafonné, connexion ouverte en lecture seule.
- Il complète le RAG texte sans le remplacer : FAISS reste utilisé pour les documents (Reddit/PDF), tandis que SQLite répond aux questions chiffrées.

## Mode expérimental : SQL généré par le LLM

En complément du mode contrôlé, un mode **expérimental** permet de tester l'approche « SQL généré par le LLM » demandée dans le cadrage, **sans casser le mode actuel**. Il sert uniquement à comparer trois conditions sur les questions chiffrées :

1. **`controlled_sql`** — mode actuel : intentions + requêtes SQL contrôlées (`utils/sql/nba_intents.py`) ;
2. **`controlled_hybrid`** — mode actuel hybride : chiffre SQL contrôlé + rédaction LLM ;
3. **`llm_sql`** — le LLM génère une requête SQL à partir de la question, puis cette requête passe par le **SQL Tool sécurisé** existant.

Le choix se fait par configuration, le défaut restant le mode contrôlé :

```env
SQL_GENERATION_MODE=controlled   # défaut : mode de production, inchangé
SQL_GENERATION_MODE=llm          # expérimental : SQL généré par le LLM
```

Garde-fous (non négociables) :

- le **LLM n'exécute jamais** de SQL : il propose seulement un texte de requête (`utils/sql/llm_sql_generator.py`), avec une sortie structurée validée par Pydantic (`should_query`, `sql`, `reason`, `expected_result_type`) ;
- toute requête générée est **revalidée** (lecture seule : `SELECT`/`WITH` uniquement, une seule requête, mots-clés d'écriture interdits) puis **exécutée par le SQL Tool sécurisé** en lecture seule (`mode=ro`) — aucune écriture en base n'est possible ;
- une **limite de lignes** est imposée à l'exécution, même si le LLM l'oublie ;
- si la question n'est pas couverte par le schéma (donnée absente, hors NBA, ambiguë), le LLM répond `should_query=false` et l'assistant refuse honnêtement.

Pour **analyser les requêtes générées** (et non les scores RAGAS), un script dédié exécute le pipeline LLM→SQL sur un panel de questions variées (simples, classements, stats joueur, totaux équipe, non supportées, hors NBA, ambiguës, dangereuses) et trace pour chacune la requête, sa validation, son exécution et un aperçu du résultat :

```bash
SQL_GENERATION_MODE=llm poetry run python scripts/analyze_llm_sql_generation.py
```

Résultats écrits dans `evaluation/results/` : `llm_sql_generation_analysis.csv` et `llm_sql_generation_analysis.json`.

La comparaison RAGAS des trois conditions reste possible via le même pipeline (`scripts/evaluate_ragas.py`), en activant `SQL_GENERATION_MODE=llm`, mais n'est pas lancée automatiquement (coût API).

