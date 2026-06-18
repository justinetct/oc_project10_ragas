# Rapport de mise en place et d'évaluation du système RAG

## Sommaire

1. [Contexte du projet](#1-contexte-du-projet)
2. [Dataset utilisé](#2-dataset-utilisé)
3. [Méthodologie d'évaluation](#3-méthodologie-dévaluation)
   - [Jeu de questions](#jeu-de-questions)
   - [Métriques](#métriques)
   - [Tests de robustesse](#tests-de-robustesse)
4. [RAG v1 — baseline](#4-rag-v1--baseline)
   - [Audit du prototype initial](#audit-du-prototype-initial)
   - [Évaluation RAGAS](#évaluation-ragas)
5. [RAG v2 — contrôlé](#5-rag-v2--contrôlé)
   - [Contrôle des données](#contrôle-des-données)
      - [Pydantic](#pydantic)
      - [Pydantic AI](#pydantic-ai)
      - [Logfire](#logfire)
   - [Évaluation](#évaluation-de-rag-v2)
      - [Comparaison avant / après](#comparaison-avant--après)
      - [Robustesse des résultats](#robustesse-des-résultats)
      - [Limites restantes](#limites-restantes)
7. [RAG v3 — hybride SQL](#7-passage-à-rag-v3--hybride-sql)
   - [Architecture générale](#architecture-générale)
   - [Données structurées et ingestion](#données-structurées-et-ingestion)
   - [Routage et sécurité](#routage-et-sécurité)
   - [Résultats et choix retenus](#résultats-et-choix-retenus)
   - [Expérimentation LLM→SQL](#expérimentation-llmsql)
8. [Limites, biais et risques](#8-limites-biais-et-risques)
9. [Conclusion](#9-conclusion)
10. [Annexes](#10-annexes)
    - [Annexe A — schéma SQLite détaillé](#annexe-a--schéma-sqlite-détaillé)
    - [Annexe B — exemple d'appel du SQL Tool](#annexe-b--exemple-dappel-du-sql-tool)
    - [Annexe C — exemples de requêtes SQL](#annexe-c--exemples-de-requêtes-sql)
    - [Annexe D — résultats détaillés du mode LLM→SQL](#annexe-d--résultats-détaillés-du-mode-llmsql)

---

## 1. Contexte du projet

L'application est un assistant conversationnel sur la NBA. Elle permet d'interroger des discussions de fans, des rapports et des statistiques de saison.

Elle repose sur un pipeline RAG (*Retrieval-Augmented Generation*). Avant de répondre, le système cherche des passages pertinents dans les documents. Il demande ensuite à un modèle de langage de rédiger une réponse à partir de ces passages.

Les sources sont mixtes : quatre PDF de discussions Reddit et un fichier Excel de statistiques de saison (`regular NBA.xlsx`).

Le travail s'organise autour d'une progression en trois versions :

1. **RAG v1 — baseline** : auditer le prototype initial et mesurer ses limites avec RAGAS ;
2. **RAG v2 — contrôlé** : sécuriser le pipeline avec Pydantic, Pydantic AI et Logfire, puis vérifier si l'ancrage des réponses progresse ;
3. **RAG v3 — hybride SQL** : ajouter une couche SQLite et un routage RAG / SQL / hybride / refus pour mieux traiter les questions chiffrées, tout en documentant les limites, les risques et les choix non retenus.

Pour simplifier la lecture, les versions sont nommées ainsi :

| Nom | Description | Tag Git |
|---|---|---|
| **RAG v1 — baseline** | pipeline RAG initial | `rag-v1-baseline` |
| **RAG v2 — contrôlé** | ajout de Pydantic, Pydantic AI et Logfire | `rag-v2-controlled` |
| **RAG v3 — hybride SQL** | ajout de SQLite et du SQL Tool pour les questions chiffrées | `rag-v3-sql-hybrid` |


---

## 2. Dataset utilisé

Le corpus combine deux types de données complémentaires :

- **PDF Reddit** : quatre PDF de discussions de fans NBA, issus de captures d'écran. Le texte extrait peut contenir du bruit lié aux captures / OCR. Ces documents servent aux questions d'opinion, de synthèse et de discussion ;
- **fichier Excel NBA** : statistiques agrégées de saison. Il contient plusieurs feuilles :
  - `Données NBA` : table principale, avec 569 joueurs et 45 colonnes de statistiques joueur-saison (`Player`, `Team`, `Age`, `GP`, `PTS`, `REB`, `AST`, `3P%`, `TS%`, `USG%`, `PIE`…). C'est la source des requêtes chiffrées ;
  - `Equipe` : référentiel des 30 équipes, avec le code à trois lettres (`BOS`, `DEN`, `LAL`…) et le nom complet ;
  - `Analyse` : synthèses déjà présentes dans le fichier, notamment le nombre de joueurs et le total de points par équipe, puis un top 15 des joueurs au nombre de points. Ces blocs sont conservés comme texte d'analyse ;
  - `Analyse Vide` : modèle de feuille d'analyse partiellement vide, conservé dans le fichier mais non utilisé comme source principale ;
  - `Dictionnaire des données` : définition des colonnes statistiques (`GP`, `PTS`, `FG%`, `OREB`, `DREB`, `AST`, `TOV`, `PIE`, etc.).

> Le fichier Excel ne contient pas de matchs individuels : les statistiques sont agrégées au niveau joueur-saison. 
> Les questions sur les 5 derniers matchs ou le découpage domicile / extérieur doivent donc être refusées ou reformulées avec une alternative sur la saison. 



---

## 3. Méthodologie d'évaluation

Pour mesurer le comportement du prototype, une évaluation automatique a été mise en place avec RAGAS. Le script `scripts/evaluate_ragas.py` exécute le vrai pipeline de l'application sur chaque question, puis calcule les métriques. Les résultats sont écrits dans `evaluation/results/`.

### Jeu de questions

Un jeu de 15 questions est défini dans `evaluation/evaluation_questions.csv`. Chaque ligne contient la question, sa catégorie, le comportement attendu et une réponse de référence courte. Un champ `requires_sql_future` marque les questions qui demandent un calcul.

Les catégories couvrent des cas variés : simple (2), complexe (2), chiffrée (5), mixte (2), bruitée (2), hors sujet (2). Une fois la première évaluation calculée, le jeu de questions n'est plus modifié.

Des exemples par catégorie :

| Catégorie | Exemple | Ce que le cas teste |
|---|---|---|
| simple | « Pour quelle équipe joue Nikola Jokić d'après les données de la saison ? » | Lecture directe d'une information présente dans les contextes. |
| complexe | « D'après les discussions Reddit, quels arguments pour et contre le tournoi play-in les fans avancent-ils ? » | Synthèse de plusieurs passages, sur des chunks bruités. |
| chiffrée | « Quel joueur a le meilleur pourcentage à 3 points (3P%) cette saison ? » | Calcul d'un maximum — cas d'hallucination observé à l'audit. |
| mixte | « Quel joueur a délivré le plus de passes décisives, et qu'est-ce que cela révèle de son rôle ? » | Un chiffre à trouver, puis une interprétation. |
| bruitée | « kl vs okc stts rebnd lst 5 gm?? » | Robustesse face à une question mal écrite. |
| hors sujet | « Quelle est la recette de la ratatouille ? » | Garde-fou : le refus est attendu. |

### Métriques

RAGAS mesure la qualité des réponses d'un RAG. Quatre métriques sont utilisées, chacune entre 0 et 1 :

- **faithfulness** : mesure le risque d'hallucination. Une réponse est mieux notée si elle reste fidèle aux contextes récupérés, et moins bien notée si elle ajoute des informations qui ne sont pas dans les sources ;
- **answer_relevancy** : mesure si la réponse répond bien à la question posée ;
- **context_precision** : en général, la précision mesure la part des éléments retournés qui sont réellement pertinents. Ici, elle mesure si les contextes récupérés par FAISS sont utiles pour répondre à la question, ou s'ils ajoutent du bruit ;
- **context_recall** : en général, le rappel mesure la part des éléments attendus qui ont bien été retrouvés. Ici, il mesure si les contextes récupérés couvrent les informations nécessaires pour construire la réponse attendue.

> Le juge qui attribue ces scores est lui-même un modèle de langage (`mistral-large-latest`). Ses scores varient d'un run à l'autre, même sans changer le pipeline. 

### Tests de robustesse

En plus des scores RAGAS, des tests automatiques (sans appel API) vérifient que le système réagit correctement aux cas problématiques, chacun rattaché à un risque métier :

- **questions bruitées** (fautes de frappe, abréviations) : la catégorie `bruitee` du jeu figé vérifie qu'une question mal écrite reste comprise (`evaluation/evaluation_questions.csv`) ;
- **questions hors sujet** : un refus poli est attendu plutôt qu'une réponse hasardeuse (`tests/test_router.py`) ;
- **SQL dangereux ou en écriture** (`DROP`, `UPDATE`, requêtes multiples…) : refusé avant toute exécution (`tests/test_sql_tool.py`, `tests/test_llm_sql_generator.py`) ;
- **données absentes** (domicile/extérieur, 5 derniers matchs) : l'assistant signale la limite et propose une alternative sur la saison au lieu d'inventer (`tests/test_router.py`) ;
- **homonymes / formulations ambiguës** : un nom de famille ambigu n'est pas résolu au hasard (`tests/test_router.py`).

Le jeu étendu `evaluation/evaluation_questions_sql_extended.csv` complète ces cas pour l'analyse plus fine des modes SQL.

---

## 4. RAG v1 — baseline

### Audit du prototype initial

L'audit complet est disponible dans `docs/audit_initial.md` et `notebooks/audit.ipynb`.

#### Fonctionnement du prototype

Le prototype suit ce flux :

```mermaid
flowchart LR
    A["Documents<br/>4 PDF + 1 Excel"] --> B["Extraction texte<br/>PDF / Excel"]
    B --> C["Découpage<br/>en chunks"]
    C --> D["Embeddings<br/>mistral-embed"]
    D --> E[("Index FAISS<br/>302 chunks")]
    E --> R["Recherche vectorielle<br/>top 5 chunks"]
    Q(["Question<br/>utilisateur"]) --> R
    R --> G["Génération<br/>mistral-small"]
    G --> S(["Réponse<br/>(Streamlit)"])
```

Étapes :

- **Extraction** : le texte est extrait des PDF Reddit et du fichier Excel, avec un bruit possible lié aux captures / OCR.
- **Découpage en chunks** : les documents sont découpés en morceaux d'environ 1 500 caractères, plus faciles à rechercher.
- **Embeddings** : chaque chunk est transformé en vecteur numérique qui représente son sens (modèle `mistral-embed`).
- **Index FAISS** : les vecteurs sont stockés dans un index de recherche rapide.
- **Recherche vectorielle** : pour chaque question, les 5 chunks les plus proches sont récupérés.
- **Génération** : un modèle Mistral (`mistral-small-latest`) rédige la réponse à partir de la question et des chunks.

L'interface est une application Streamlit (`MistralChat.py`). 

#### Ce qui fonctionne

- le pipeline tourne de bout en bout ;
- l'index FAISS est construit (302 chunks à partir de 5 documents) ;
- la recherche vectorielle retourne des contextes ;
- l'application répond aux questions.

#### Limites observées

- les réponses peuvent être peu ancrées dans les sources : le prompt initial (« animer le débat ») n'oblige pas le modèle à se limiter aux contextes ;
- FAISS retrouve du texte proche, mais ne calcule rien : il ne sait pas trouver un maximum dans le fichier Excel ;
- le texte extrait des PDF Reddit peut contenir du bruit lié aux captures / OCR ;
- les questions hors sujet ne sont pas toujours refusées (testé avec une recette de cuisine : le modèle répond souvent).

#### Exemple de limite

À la question « quel joueur a le meilleur pourcentage à 3 points ? », le modèle peut répondre **Shai Gilgeous-Alexander — 37,5 %**. Pourtant, un extrait récupéré contient déjà **Nikola Jokić — 41,7 %**.

Le système reformule un passage. Il ne calcule pas le maximum de la colonne. Le RAG seul ne suffit donc pas pour les questions de calcul.

![baseline_streamlit.png](img/baseline_streamlit.png)

### Évaluation RAGAS

Cette première mesure sert de point de départ : elle évalue le prototype tel qu'il existe au départ, sans modification du prompt, du modèle ou de la récupération FAISS.

| Métrique | Score | Lecture |
|---|---:|---|
| `faithfulness` | 0,2512 | Risque d'hallucination élevé : le modèle ajoute ou reformule trop souvent au-delà des sources. |
| `answer_relevancy` | 0,5760 | Pertinence correcte : la réponse semble souvent répondre à la question, même quand elle est mal justifiée. |
| `context_precision` | 0,3622 | Précision faible côté récupération : une partie des contextes récupérés n'aide pas vraiment à répondre. |
| `context_recall` | 0,4000 | Rappel faible côté récupération : les contextes ne couvrent pas toujours toutes les informations attendues. |

Le résultat principal est la faiblesse de la `faithfulness`. Le prototype répond, mais il s'autorise trop souvent à compléter ou reformuler au-delà des sources. Cette limite est particulièrement visible sur les questions chiffrées : FAISS peut retrouver un extrait proche, mais il ne sait pas calculer un maximum, un classement ou une moyenne.

Cette baseline fixe donc deux objectifs pour les versions suivantes : mieux encadrer la génération, puis ajouter une brique structurée pour les calculs sur les statistiques NBA.

---

## 5. RAG v2 — contrôlé

Trois briques sont ajoutées, sans changer le modèle de génération ni la récupération FAISS : 
- validation des données avec Pydantic, 
- génération structurée avec Pydantic AI, 
- traçage optionnel avec Logfire.

### Contrôle des données

#### Pydantic

>Pydantic est une librairie de validation : on décrit la forme attendue d'un objet, elle vérifie que les données la respectent.

Les modèles de `utils/schemas.py` valident les objets du pipeline :

- les **documents** chargés (texte non vide, source connue) ;
- les **chunks** indexés (identifiant, texte, métadonnées) ;
- les **contextes récupérés** (texte, score, source) ;
- la **réponse finale** (réponse non vide, contextes utilisés).

Les erreurs de structure sont ainsi détectées tôt, au lieu de se propager dans le pipeline.

#### Pydantic AI

> Pydantic AI encadre les appels au modèle : on déclare le type de sortie attendu, et la sortie est validée avec Pydantic. 
> La génération est centralisée sous forme d'un agent Pydantic AI.

L'agent utilise Mistral (`mistral-small-latest`, température 0,1, même prompt que le prototype), reçoit la question et les contextes FAISS, et produit une sortie typée `RagAnswerOutput` (réponse non vide).

Point important : `MistralChat.py` (l'application) et `scripts/evaluate_ragas.py` (l'évaluation) utilisent **le même agent**. L'évaluation mesure donc le même chemin de génération que l'application.

Pydantic AI ne rend pas le modèle meilleur en soi. Il rend la génération plus structurée : sortie au format garanti, validation systématique, code unique. Si le modèle renvoie une réponse vide, l'agent lève une erreur ; le script d'évaluation gère ce cas avec quelques ré-essais.

#### Logfire

> Logfire est un outil de traçage : il enregistre ce qui se passe pendant l'exécution et l'affiche dans une interface web.

Son intégration est optionnelle et non bloquante :

- **sans token** (clé d'accès), l'application fonctionne normalement, rien n'est envoyé ;
- **avec token**, les étapes clés sont tracées : recherche vectorielle, génération (un span par question, appels Mistral inclus), évaluation RAGAS.

Cela aide à comprendre le comportement du pipeline : temps passé par étape, erreurs API, contextes récupérés.

![baseline_logfire.png](img/baseline_logfire.png)

---

### Évaluation de RAG v2

Après le passage à `RAG v2 — contrôlé`, l'évaluation a été relancée : mêmes questions, mêmes métriques, même juge. La génération passe par l'agent Pydantic AI ; la récupération FAISS est inchangée.

#### Comparaison avant / après

| Métrique | Avant | Après | Évolution | Lecture |
|---|---:|---:|---:|---|
| `faithfulness` | 0,2512 | 0,3534 | +0,1022 | Le risque d'hallucination baisse : les réponses sont mieux alignées avec les contextes. |
| `answer_relevancy` | 0,5760 | 0,5520 | −0,0240 | La pertinence directe baisse légèrement : le modèle répond de façon un peu moins libre, mais plus contrôlée. |
| `context_precision` | 0,3622 | 0,3622 | 0,0000 | La précision de récupération ne change pas, ce qui est attendu car FAISS est inchangé. |
| `context_recall` | 0,4000 | 0,4333 | +0,0333 | Le rappel progresse légèrement, mais l'effet reste limité : la récupération n'est pas la brique modifiée. |

L'amélioration vient donc surtout de la génération : le même contexte est utilisé, mais la réponse est mieux encadrée.

Le juge varie d'un run à l'autre. Pour éviter de conclure sur un seul score, la sous-section suivante répète l'évaluation plusieurs fois.

#### Robustesse des résultats

L'évaluation a été relancée 5 fois pour chaque version du pipeline, dans les mêmes conditions. Le tableau ci-dessous se concentre sur la `faithfulness`, car c'est la métrique qui mesure le mieux le risque d'hallucination.

| Indicateur | Avant (ancien pipeline) | Après (agent Pydantic AI) | Lecture |
|---|---:|---:|---|
| run 0 | 0,251 | 0,353 | Le premier run retrouve le gain observé dans le tableau précédent. |
| run 1 | 0,238 | 0,334 | Le score reste supérieur après passage par l'agent. |
| run 2 | 0,188 | 0,393 | Le plus mauvais run de l'ancien pipeline reste loin du niveau atteint après correction. |
| run 3 | 0,289 | 0,391 | Même le meilleur score de l'ancien pipeline reste sous les meilleurs scores de RAG v2. |
| run 4 | 0,278 | 0,310 | Le gain existe encore, même sur le run le moins favorable après correction. |
| **Moyenne** | **0,249** | **0,356** | Gain moyen d'environ +0,10 : le risque d'hallucination baisse de façon répétée. |
| **Étendue** | 0,188–0,289 | 0,310–0,393 | Les deux plages ne se recouvrent pas : l'effet dépasse la simple variation du juge. |
| **Écart-type** | 0,040 | 0,036 | La variabilité est comparable avant et après ; le gain ne vient donc pas d'un run isolé. |

> La hausse de `faithfulness` apparaît donc comme une **tendance nette** : les réponses sont mieux encadrées avec l'agent Pydantic AI. En contrepartie, l'`answer_relevancy` moyenne baisse (≈ 0,50 contre ≈ 0,65) : les réponses sont plus proches des sources, mais parfois moins directes. Avec 5 + 5 runs, l'interprétation reste prudente : le signal est clair, sans être une preuve absolue.

#### Limites restantes

- le RAG reste fragile pour les questions qui demandent un calcul exact (maximum, moyenne, classement) ;
- FAISS cherche du texte proche, il ne calcule pas : les données Excel demandent une brique structurée ;
- les PDF OCR peuvent contenir du bruit ;
- les questions hors sujet ne sont pas systématiquement refusées.

---

## 7. RAG v3 — hybride SQL

Cette version complète le RAG texte avec un accès structuré aux statistiques. Le fichier Excel est chargé dans une base SQLite, interrogée avec un SQL Tool LangChain en lecture seule. Le routeur oriente chaque question vers le bon traitement : RAG texte, SQL, réponse hybride ou refus hors périmètre.

### Architecture générale

L'architecture se lit en deux temps : la préparation des données (hors ligne), puis le traitement d'une question.

**Préparation des données.** Les documents texte (PDF Reddit) sont découpés, vectorisés (`mistral-embed`) et stockés dans FAISS. En parallèle, le fichier Excel est chargé — validé par Pydantic — dans une base SQLite dédiée aux statistiques.

![Indexation des données](img/architecture_indexation.png)

**Traitement d'une question.** Le routeur choisit, par règles de mots-clés, l'une des quatre routes, et chacune produit la réponse affichée dans Streamlit. Les questions documentaires passent par FAISS, les questions chiffrées par SQLite en lecture seule, les questions mixtes par les deux (chiffre vérifié + rédaction), et les questions hors NBA reçoivent un refus poli.

![Traitement d'une question (routage)](img/architecture_routage.png)


### Données structurées et ingestion

Le passage à SQL ne transforme pas toute l'application en base de données. Il ajoute seulement une couche structurée pour les statistiques qui demandent un calcul exact. Cinq tables structurent ces données :

- `teams` : référentiel des équipes (justifié par la feuille `Equipe`) ;
- `players` : les joueurs, rattachés à leur équipe ;
- `matches` : table minimaliste — elle représente le périmètre de la saison régulière, sans inventer de matchs ;
- `stats` : les statistiques de saison par joueur ;
- `reports` : les blocs d'analyse textuels de la feuille `Analyse`.

Le schéma détaillé de la base est donné en [annexe A](#annexe-a--schéma-sqlite-détaillé). Dans le corps du rapport, l'important est surtout le choix de séparation : référentiel équipes, référentiel joueurs, statistiques chiffrées et blocs d'analyse textuels.

Trois points repérés à l'analyse et traités dès l'import :

- la colonne `3PM` est interprétée par Excel comme une heure (`15:00:00`) : elle est renommée à la lecture ;
- les classements par pourcentage demandent un filtre de volume (au moins 100 tentatives), sinon un joueur à 1 tir réussi sur 1 apparaît à 100 % ;
- les pourcentages sont stockés tels quels (`37.5` pour 37,5 %), sans conversion.

#### Ingestion et contrôles

Le pipeline est : `Excel → validation Pydantic → SQLite → requêtes de contrôle`. 

Chaque ligne est validée avant insertion (`utils/sql/schemas.py`). La base est générée localement :

```bash
poetry run python scripts/load_excel_to_db.py
```

**Contrôles sur la base générée :** 
- 30 équipes, 
- 569 joueurs, 
- 569 lignes de statistiques, 
- 1 ligne `matches`, 
- 2 rapports ; 
- 0 ligne orpheline, 
- 0 violation de clé étrangère.
- Meilleur marqueur : Shai Gilgeous-Alexander (2 485 points). 
- Meilleur 3P% avec au moins 100 tentatives : Seth Curry (45,6 %).

### Routage et sécurité

#### SQL Tool LangChain en lecture seule

Le module `utils/sql/sql_tool.py` expose l'outil qui interroge cette base, en deux niveaux :

- une fonction interne sécurisée, `run_read_only_query()`, qui contrôle puis exécute la requête ;
- un tool LangChain (`StructuredTool`), nommé `nba_sql_query`, avec un schéma d'entrée Pydantic (`query`, `params`, `limit`) branché sur l'assistant.

Règles de sécurité (lecture seule stricte) :

- seules les requêtes `SELECT` (ou `WITH`) sont acceptées ;
- les mots-clés d'écriture sont refusés (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `PRAGMA`…) ;
- une seule requête à la fois : `SELECT …; DROP TABLE …` est refusé ;
- le nombre de lignes retournées est plafonné (20 par défaut) ;
- la connexion SQLite est ouverte en mode lecture seule : même une requête qui passerait les filtres ne pourrait pas écrire.

Exemples de questions qui passeront par SQL :

- « Quel joueur a marqué le plus de points ? »
- « Qui a le meilleur pourcentage à 3 points avec au moins 100 tentatives ? »
- « Combien de joueurs par équipe ? »
- « Quelles sont les stats de Nikola Jokić ? »

Les questions sur les documents (« que disent les fans sur les Lakers ? ») restent côté RAG texte.

Le tool renvoie des données structurées ; l'assistant rédige ensuite la réponse. Un exemple d'appel est donné en [annexe B](#annexe-b--exemple-dappel-du-sql-tool).

Le module est couvert par 20 tests sans appel API (`tests/test_sql_tool.py`) : requêtes valides, refus d'écriture, plafond de lignes, erreurs propres, appel du tool via `.invoke(...)`.

#### Routage intégré : l'assistant choisit son chemin

Le module `utils/router.py` classe chaque question vers quatre chemins, utilisés à la fois par l'application et par l'évaluation :

| Route | Cas | Traitement |
|---|---|---|
| `rag` | documents, opinions, discussions Reddit | FAISS + génération |
| `sql` | question chiffrée | requête contrôlée via le SQL Tool |
| `hybrid` | chiffre + interprétation | SQL d'abord, puis rédaction LLM |
| `out_of_scope` | hors NBA / hors sources | refus poli, aucun appel externe |

Pour les questions chiffrées, le mode par défaut (`controlled`) s'appuie sur des cas connus : classement, statistique joueur, filtre équipe ou seuil numérique. Une requête SQL prédéfinie est ensuite exécutée en lecture seule par le SQL Tool.

Si la donnée demandée n'existe pas dans la base, le système refuse de calculer et explique la limite. L'application et l'évaluation utilisent la même fonction `answer_question()`, donc les scores RAGAS mesurent le même comportement que l'interface.

Le jeu figé E01–E15 reste la référence officielle. Le jeu étendu SQL sert seulement à analyser plus finement les cas chiffrés.

### Résultats et choix retenus

#### Avant / après routage
Trois conditions sont comparées sur le jeu figé E01–E15, avec les mêmes métriques et le même juge que les évaluations précédentes : RAG texte seul avant routage SQL, routage avec hybride `sql_only`, routage avec hybride `sql_with_rag_context`.

Comme le juge RAGAS varie d'un run à l'autre, chaque condition a été relancée 5 fois. Le tableau ci-dessous présente les scores moyens.

| Condition | faithfulness | answer_relevancy | context_precision | context_recall | Lecture |
|---|---:|---:|---:|---:|---|
| RAG texte seul (avant routage SQL) | 0,356 | 0,504 | 0,423 | 0,407 | Point de comparaison : le RAG seul reste fragile sur les questions chiffrées. |
| Routage · hybride `sql_only` | 0,509 | 0,629 | 0,547 | 0,638 | Meilleur compromis : les réponses chiffrées s'appuient sur un fait SQL exact. |
| Routage · hybride `sql_with_rag_context` | 0,505 | 0,637 | 0,535 | 0,624 | Variante testée, sans gain clair par rapport à `sql_only`. |

Le routage améliore les quatre métriques, surtout la pertinence (`answer_relevancy` 0,50 → 0,63–0,64) et la récupération de contexte. Le gain vient principalement des questions chiffrées : elles ne dépendent plus d'extraits approximatifs, mais d'une requête SQL.

Lecture par catégorie (`faithfulness`) :

| Catégorie | RAG texte seul | Routage SQL / hybride | Lecture |
|---|---:|---:|---|
| Chiffrées | 0,50 | 0,84–0,85 | Gain attendu : le SQL renvoie la valeur exacte là où le RAG approximait ou hallucinait. |
| Bruitées | 0,05 | 0,50–0,52 | Les formulations imparfaites comme « meilleur tireur a 3pts cette saion?? » sont mieux reconnues comme des questions chiffrées. |
| Mixtes | 0,12 | 0,16–0,19 | Gain limité : la partie interprétative reste difficile à justifier avec des citations exactes. |
| Hors-sujet | 0,23 | 0,00 | Régression apparente : le refus poli est voulu, mais RAGAS le note mal car il ne cite aucun contexte. |

La distribution des routes sur E01–E15 confirme le comportement attendu : 5 `rag`, 6 `sql`, 2 `hybrid`, 2 `out_of_scope`.

#### Choix du mode hybride

Les deux variantes ne changent que les questions `hybrid` : `sql_only` rédige à partir du seul chiffre SQL ; `sql_with_rag_context` ajoute quelques extraits FAISS pour la partie qualitative, avec une consigne explicite : les chiffres SQL font foi.

Les écarts observés restent limités par rapport au bruit du juge RAGAS. 

> Le mode `sql_only` est donc retenu par défaut : il est plus simple, plus stable et moins coûteux. 

Le mode `sql_with_rag_context` reste disponible par configuration (`HYBRID_MODE=sql_with_rag_context`) pour les cas où l'on veut enrichir l'interprétation avec des contextes texte.

### Expérimentation LLM→SQL

Un mode `llm_sql` a aussi été testé : le LLM propose une requête SQL, puis le SQL Tool la valide et l'exécute en lecture seule. Le LLM n'a donc jamais accès directement à la base.

Ce mode améliore la souplesse sur certaines formulations, mais il n'est pas retenu comme choix final. Le mode contrôlé reste plus stable, déterministe et plus facile à vérifier. Le projet garde donc `controlled` + `sql_only` par défaut.

Les résultats détaillés de cette comparaison sont placés en [annexe D](#annexe-d--résultats-détaillés-du-mode-llmsql).

#### Figures de synthèse (preuves visuelles)

Les figures ci-dessous consolident les résultats discutés plus haut. Elles sont régénérées **sans appel API** à partir de `evaluation/results/variance_runs/` (`poetry run python scripts/make_report_figures.py`). **Chaque mode — baseline RAG incluse — est moyenné sur ses 5 runs de variance** (barres d'erreur = écart-type inter-runs) : sur un run unique, des moyennes proches coïncidaient par hasard — par exemple un `context_recall` identique sur les trois modes routés — alors que la moyenne sur 5 runs les sépare. Les petits écarts avec les tableaux ci-dessus relèvent de la variance du juge entre runs.

**Scores globaux par mode** — repère d'ensemble (baseline RAG → SQL contrôlé → hybride → LLM→SQL). Ces scores doivent être lus avec les tableaux précédents, car le jeu figé mélange des questions documentaires, chiffrées, mixtes et hors sujet.

> À noter : `controlled_hybrid` ne diffère de `controlled_sql` que sur les 2 questions hybrides (E05, E06) — sur les 13 autres, le contexte récupéré est identique ; l'écart global entre ces deux barres tient donc surtout au bruit du juge, pas à un effet large du « + contexte RAG ».

![Scores RAGAS globaux par mode](img/ragas_global_scores.png)

**Gains vs baseline RAG, hors questions hors sujet** — l'apport du SQL par métrique. Le gain de fidélité et de contexte est net, concentré sur les questions chiffrées et bruitées (détail par catégorie dans le notebook).

![Gains RAGAS vs baseline RAG, hors questions hors sujet](img/ragas_gains_vs_baseline.png)

**Route SQL — contrôlé vs LLM→SQL (moyenne ± écart-type, 5 runs)** — le duel central. `llm_sql` atteint la parité en `faithfulness` ; le contrôlé est plus stable (écart-type ≈ 0) et reste le mode par défaut.

![Route SQL — contrôlé vs LLM→SQL, 5 runs](img/ragas_sql_route_x5.png)

**Métriques complémentaires** — `aspect_critic` = 1,0 pour les deux modes (aucune statistique absente inventée) ; `answer_correctness` reste modérée (≈ 0,50), à lire avec prudence.

![Métriques complémentaires — answer_correctness et aspect_critic](img/ragas_extra_metrics.png)

#### Limites du routage actuel

- routage par règles : robuste sur le jeu testé, mais une question très déformée peut être mal classée (E12 reste en `rag`) ;
- en mode contrôlé (défaut), la couverture SQL est bornée aux requêtes prédéfinies ; le mode `llm_sql` couvre davantage de questions composées. Ce qui est absent de la base (ex. splits domicile/extérieur) reste signalé comme indisponible dans les deux modes ;
- RAGAS juge mal deux familles de réponses correctes : les refus (hors-sujet, note 0 par construction) et les réponses interprétatives des questions mixtes ; le comparatif chiffré est donc complété d'une lecture qualitative des réponses, en particulier sur les questions mixtes.

## 8. Limites, biais et risques

Plusieurs limites restent à garder en tête.

- **Données NBA** : le fichier Excel contient des statistiques agrégées sur la saison. Il ne contient pas de matchs individuels, de 5 derniers matchs ni de découpage domicile / extérieur. Les questions de ce type doivent donc être refusées ou reformulées avec une alternative sur la saison.
- **Routage et SQL** : le mode contrôlé est stable mais limité aux intentions prévues. Le mode `llm_sql` couvre davantage de formulations, mais il reste plus variable et peut produire une requête valide mais mal adaptée à la question. C’est pour cela que toutes les requêtes passent par validation et lecture seule.
- **Corpus Reddit** : les PDF viennent de captures et de texte extrait avec un bruit possible lié à l'OCR. Ils contiennent des fautes et des opinions de fans. Le système peut résumer ces discussions, mais elles ne représentent pas toute la NBA.
- **Évaluation RAGAS** : le jeu figé contient 15 questions, dont peu de questions SQL. Le juge LLM varie d’un run à l’autre et note mal certains refus pourtant corrects. Les résultats sont donc lus comme des tendances, avec des runs répétés quand c’est nécessaire.
- **Généralisation** : les résultats valent pour ce corpus, ce modèle et ces données. Un changement de modèle, de saison NBA ou de documents demanderait de relancer l’évaluation.

Les pistes d’amélioration réalistes seraient d’ajouter des données match par match, d’élargir le jeu de questions, de surveiller les cas mal routés et de relancer les métriques après tout changement important de modèle ou de corpus.

---

## 9. Conclusion

Le prototype RAG fonctionne : il indexe les documents, retrouve des contextes et répond aux questions.

L'évaluation RAGAS a montré ses limites : ancrage faible (`faithfulness` à 0,25) et questions chiffrées fragiles.

RAG v2 — contrôlé renforce le pipeline avec Pydantic, Pydantic AI et Logfire. Après ces changements, la `faithfulness` moyenne passe de 0,25 à 0,36 sur 5 runs, avec une baisse de la pertinence directe. C'est une tendance à lire avec prudence : le juge varie d'un run à l'autre.

RAG v3 — hybride SQL utilise un routage en quatre chemins : RAG texte pour le documentaire, requêtes SQL prédéfinies pour le chiffré, réponse hybride pour les questions mixtes, refus hors périmètre. Sur le jeu figé E01–E15, le routage fait passer la `faithfulness` moyenne de 0,36 à 0,51 et la pertinence des réponses de 0,50 à 0,63–0,64, avec des gains concentrés exactement là où le RAG seul échouait : questions chiffrées et questions bruitées à intention chiffrée.

Sur les questions chiffrées, le SQL améliore donc nettement le RAG seul. Le mode contrôlé est conservé par défaut : plus stable, déterministe et plus facile à vérifier. Les limites des données sont signalées plutôt que comblées par des chiffres inventés (`aspect_critic` = 1,0 dans les comparaisons complémentaires). Le mode `llm_sql`, détaillé en annexe, confirme qu'une génération SQL par le modèle est possible, mais reste expérimental.

Une variante de prompt plus strict est aussi disponible (`RAG_PROMPT_MODE=strict`). Elle améliore l'ancrage des réponses, mais réduit la pertinence sur les questions de discussion. Le prompt prototype reste donc le défaut, et le prompt strict reste activable par configuration.

La prochaine étape serait d'élargir la couverture des questions chiffrées et de remplacer le routage par règles par un classifieur plus robuste si le besoin apparaît à l'usage, en gardant le même principe de sécurité : toute requête, contrôlée ou générée, passe par le SQL Tool en lecture seule.

> **Ce qui est exclu du périmètre**
> - pas de fine-tuning du modèle ;
> - pas de changement du modèle principal de génération ;
> - pas de modification du jeu de questions après la baseline ;

---

## 10. Annexes

### Annexe A — schéma SQLite détaillé

```mermaid
erDiagram
    TEAMS ||--o{ PLAYERS : has
    PLAYERS ||--o{ STATS : has
    MATCHES ||--o{ STATS : contains
    TEAMS {
        text team_code PK
        text team_name
    }
    PLAYERS {
        int player_id PK
        text player_name
        text team_code FK
        int age
    }
    MATCHES {
        int match_id PK
        text scope
        text description
    }
    STATS {
        int stat_id PK
        int player_id FK
        int match_id FK
        int points
        int rebounds
        int assists
        real three_point_pct
    }
    REPORTS {
        int report_id PK
        text report_type
        text title
        text content
        text source_sheet
    }
```

### Annexe B — exemple d'appel du SQL Tool

```python
sql_query_tool.invoke({
    "query": "SELECT p.player_name, s.points FROM stats s "
             "JOIN players p ON p.player_id = s.player_id "
             "ORDER BY s.points DESC",
    "limit": 2,
})
# [{'player_name': 'Shai Gilgeous-Alexander', 'points': 2485},
#  {'player_name': 'Anthony Edwards', 'points': 2180}]
```

### Annexe C — exemples de requêtes SQL

Quelques requêtes types sur la base `data/nba.sqlite`, pour illustrer ce que le SQL Tool exécute en lecture seule :

```sql
-- Nombre de joueurs (attendu : 569)
SELECT COUNT(*) FROM players;

-- Top 10 des marqueurs
SELECT p.player_name, p.team_code, s.points
FROM stats s JOIN players p ON p.player_id = s.player_id
ORDER BY s.points DESC LIMIT 10;

-- Meilleur 3P% avec filtre de volume (évite l'artefact d'un joueur à 1 tir sur 1)
SELECT p.player_name, s.three_point_pct, s.three_points_attempted
FROM stats s JOIN players p ON p.player_id = s.player_id
WHERE s.three_points_attempted >= 100
ORDER BY s.three_point_pct DESC LIMIT 5;
```

### Annexe D — résultats détaillés du mode LLM→SQL

Le mode `llm_sql` a été testé pour mesurer l'intérêt d'une génération de requêtes SQL par le modèle. Il ne remplace pas le mode contrôlé, mais permet de vérifier si une approche plus souple peut atteindre un niveau comparable.

Trois conditions sont comparées sur les questions chiffrées : `controlled_sql`, `controlled_hybrid` (chiffre SQL + contextes RAG sur les questions hybrides) et `llm_sql`. Le juge RAGAS étant bruité, chaque condition est lancée 5 fois (`scripts/run_all_ragas.sh`) puis moyennée (`scripts/aggregate_variance_runs.py`).

Route SQL — moyenne ± écart-type sur 5 runs :

| Métrique | SQL contrôlé | LLM→SQL |
|---|---|---|
| `faithfulness` | 0,860 ± 0,015 | 0,910 ± 0,028 |
| `answer_relevancy` | 0,653 ± 0,001 | 0,630 ± 0,004 |
| `context_precision` | 0,667 ± 0,000 | 0,667 ± 0,000 |
| `context_recall` | 0,778 ± 0,000 | 0,711 ± 0,091 |

Après correction du format des réponses, affichage d'un top 5 et ajout des exemples few-shot, le mode `llm_sql` atteint la parité avec le SQL contrôlé sur la route SQL. Le SQL contrôlé reste plus stable (écart-type proche de 0) : il est donc conservé comme mode par défaut.

`controlled_hybrid` ne modifie que les questions hybrides (ajout de contextes RAG). Les écarts avec `controlled_sql` restent limités ; `controlled_sql` est gardé comme référence par défaut, et `controlled_hybrid` reste disponible par configuration.

**Métriques complémentaires.** Deux mesures optionnelles ont été ajoutées en lecture complémentaire. `answer_correctness` reste modérée (≈ 0,50 pour les deux modes) : les refus et les limites de données sont difficiles à noter avec une réponse de référence classique. `aspect_critic` vaut 1,0 pour les deux modes, ce qui indique que les réponses respectent les limites des données — aucune statistique absente n'est inventée. Le détail figure dans `notebooks/sql_modes_analysis.ipynb`.
