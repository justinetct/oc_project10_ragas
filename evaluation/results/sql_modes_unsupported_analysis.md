# Évaluation complémentaire — questions chiffrées non supportées (contrôlé vs LLM→SQL)

> Petite évaluation ciblée, **séparée** du jeu officiel E01–E15. Objectif : sur des
> questions chiffrées **impossibles ou non couvertes** par le schéma actuel (stats
> agrégées de saison, une ligne par joueur), vérifier si chaque mode **refuse
> correctement** ou **répond à côté**, et illustrer où le LLM→SQL est plus adaptable
> que le mode contrôlé — tout en restant encadré par le SQL Tool en lecture seule.

- **Dataset** : [`evaluation/evaluation_questions_unsupported.csv`](../evaluation_questions_unsupported.csv) (5 questions, vrais noms de joueurs/équipes).
- **Comparaison brute** : [`sql_modes_unsupported_comparison.csv`](sql_modes_unsupported_comparison.csv) / `.json` (les deux modes en un passage).
- **RAGAS `aspect_critic`** : `ragas_routed_controlled_sql_only_evaluation_questions_unsupported_with_aspect_*` et `ragas_routed_llm_sql_only_evaluation_questions_unsupported_with_aspect_*`.

## Méthode

1. **Comparaison sans RAGAS** (`scripts/compare_sql_modes_extended.py --label unsupported`) :
   exécute le mode contrôlé (routeur + mapping figé) **et** le générateur LLM→SQL pour
   chaque question, et trace `should_query` / validation / exécution + un `status`.
2. **RAGAS `aspect_critic`** : métrique binaire 0/1 « la réponse respecte les limites des
   données NBA disponibles » (ne pas inventer, signaler les limites). Les 4 métriques
   classiques sont basses **par construction** sur des refus et ne sont pas l'angle ici.
3. **Lecture métier manuelle** : verdict par question × mode parmi
   `refus correct` / `repli honnête` / `réponse à côté` / `chiffre inventé` / `erreur`.

## Précautions de lecture (à ne pas masquer dans le rapport)

- **Le repli saison court-circuite le mode LLM.** Dans `utils/router.py::_answer_sql`,
  `answer_season_fallback()` est évalué **avant** le choix du mode. Quand il se déclenche
  (ex. U02), **les deux modes renvoient en production la même réponse honnête** et
  `run_llm_sql` n'est jamais appelé. La divergence contrôlé vs LLM n'apparaît donc que
  sur les questions où ce repli ne se déclenche pas (U01, U03, U04, U05).
- **Le script de comparaison appelle `run_llm_sql` en isolation** : pour U02 il montre ce
  que le générateur ferait seul (refus), pas la réponse de production (repli saison).
- **Les vrais noms sont obligatoires.** Un placeholder (« joueur X », « A et B ») fausse le
  test : « X » fait router la question en RAG, et un nom inconnu déclenche le refus
  générique au lieu de la « réponse à côté » révélatrice.

## Résultats par question

| ID | Question | Contrôlé — comportement observé | Verdict contrôlé | LLM→SQL — comportement observé | Verdict LLM |
|----|----------|----------------------------------|------------------|--------------------------------|-------------|
| U01 | meilleur 3P% **sur 5 derniers matchs** | classement 3P% **saison** (≥100 tentatives) ; perd « 5 derniers matchs » en silence | **réponse à côté** | `should_query=false`, motif « pas de match par match » | **refus correct** |
| U02 | rebonds Lakers **domicile/extérieur** | repli saison : signale l'absence + total rebonds/équipe | **repli honnête** | générateur isolé : refuse (split absent). *Prod : même repli saison que le contrôlé* | **refus correct** |
| U03 | évolution points **Jokić** 5 derniers | **fiche saison** de Jokić ; perd « 5 derniers matchs » | **réponse à côté** | `should_query=false`, motif « pas d'historique par date » | **refus correct** |
| U04 | rebonds **par match** Jokić & LeBron | fiche saison de **LeBron seul** (perd Jokić + le « par match ») | **réponse à côté** | `should_query=true` mais génère `s.rebounds_per_game` (colonne inexistante) → **bloqué par le SQL Tool** (`no such column`) → repli refus | **erreur contenue → refus** |
| U05 | le plus **progressé** | « non pris en charge » (catalogue générique) | **refus correct** (motif générique) | `should_query=false`, motif « pas de multi-saisons » | **refus correct** (motif ciblé) |

### Synthèse par mode (lecture métier)

| Verdict | Contrôlé | LLM→SQL |
|---------|:--------:|:-------:|
| refus correct | 1 (U05) | 4 (U01, U02, U03, U05) |
| repli honnête | 1 (U02) | 0 *(repli identique en prod)* |
| réponse à côté | **3 (U01, U03, U04)** | 0 |
| erreur contenue → refus | 0 | 1 (U04) |
| **chiffre inventé** | **0** | **0** |

Statuts automatiques du script (`status_counts`) : `llm_refuses=3`, `same=1`, `llm_error=1`.

### RAGAS — `aspect_critic` (respect des limites des données, 1.0 = respecte)

| ID | Contrôlé | LLM→SQL |
|----|:--------:|:-------:|
| U01 | 0.0 | 1.0 |
| U02 | 0.0 | 0.0 |
| U03 | 0.0 | 1.0 |
| U04 | 0.0 | 1.0 |
| U05 | 1.0 | 1.0 |
| **Moyenne** | **0.20** | **0.80** |

`aspect_critic` confirme la lecture manuelle : le contrôlé ne « respecte les limites »
que sur U05 (le seul cas où il refuse) ; le LLM les respecte 4/5. **U02 = 0.0 pour les
deux** car le repli saison propose des totaux de saison en substitut — le juge ne le
considère pas comme un plein respect de la limite (à mi-chemin entre refus et réponse).

#### Inversion des 4 métriques classiques (à expliquer dans le rapport)

| Métrique (moyenne) | Contrôlé | LLM→SQL |
|--------------------|:--------:|:-------:|
| faithfulness | **0.715** | 0.127 |
| answer_relevancy | **0.479** | 0.000 |
| context_precision | 0.117 | 0.000 |
| context_recall | 0.400 | 0.300 |
| **aspect_critic** | 0.200 | **0.800** |

Point méthodologique majeur : sur ces questions non supportées, les **4 métriques
classiques récompensent le contrôlé** (qui *répond*, même à côté : des chiffres réels et
ancrés dans le contexte SQL → faithfulness/relevancy élevées) et **pénalisent le LLM**
(un refus n'a ni contexte ni « réponse » à juger → scores ~0). Elles mesurent « a-t-il
répondu de façon ancrée ? », pas « **fallait-il répondre ?** ». Seul `aspect_critic`
(ou la lecture métier) capture la bonne attitude et **inverse le classement** : 0.20
contrôlé vs 0.80 LLM. C'est l'argument central pour ne PAS conclure ces cas sur les 4
métriques historiques.

## Conclusion

- **Sur les 4 cas réellement discriminants** (U01, U03, U04, U05 — le repli saison ne
  court-circuite pas), le LLM→SQL **refuse ou contient l'erreur 4/4**, alors que le
  contrôlé **répond à côté 3/4**. Le contrôlé n'est « bon » que là où une **règle a été
  écrite** (repli saison U02, catalogue de refus U05) : cela confirme qu'il **faut ajouter
  une règle à chaque évolution du schéma** pour qu'il reste pertinent.
- **Le LLM→SQL détecte les contraintes absentes du schéma** (5 derniers matchs,
  domicile/extérieur, multi-saisons) **sans règle dédiée** → plus adaptable quand le schéma
  évolue ou qu'une question imprévue arrive.
- **Les deux modes restent sûrs : 0 chiffre inventé.** Le contrôlé par construction (liste
  blanche de colonnes) ; le LLM parce que le **SQL Tool en lecture seule bloque la requête
  fautive avant tout affichage** (U04 : colonne hallucinée `rebounds_per_game` rejetée). La
  faiblesse du contrôlé n'est donc **pas l'hallucination** mais la **réponse à côté**
  (chiffres réels, mauvaise question — un cas spécial capte la question et répond sur la
  saison).
- **Limite du LLM** : il peut produire une requête syntaxiquement plausible mais erronée
  (U04) ; le garde-fou reste le SQL Tool, pas le LLM lui-même.

## Bonus — questions mieux couvertes par le LLM que par le contrôlé

- **U04 (`par match`)** : la moyenne par match est **calculable** (`rebounds / games_played`).
  Le contrôlé ne la couvre pas (il renvoie une fiche saison) ; le LLM **tente** le calcul —
  il suffirait qu'il vise la bonne expression (`CAST(s.rebounds AS REAL)/s.games_played`).
- Plus largement (déjà visible dans le jeu étendu `evaluation_questions_sql_extended.csv`) :
  filtres équipe + seuil (SQL14–21), `AVG(age)` (SQL42), ratios calculés comme
  passes/ballons perdus (SQL47), seuil + filtre de volume (SQL20). Autant de cas où écrire
  une règle contrôlée dédiée serait coûteux, et où le LLM encadré couvre plus de terrain.
