#!/usr/bin/env bash
# scripts/run_all_ragas.sh — Lance TOUS les runs RAGAS d'un coup.
#
# - variance ×REPEATS sur les 3 conditions SQL (controlled_sql / controlled_hybrid / llm_sql) ;
# - 1 passe « métriques extra » (answer_correctness + aspect_critic) par condition clé ;
# - garde un log par run + un log maître ;
# - archive chaque run dans variance_runs/ (aucun écrasement entre les répétitions) ;
# - affiche le temps écoulé et le temps restant estimé.
#
# Aucun appel API n'est fait par ce script lui-même : il appelle evaluate_ragas.py.
# Usage :
#   bash scripts/run_all_ragas.sh           # demande confirmation (affiche le plan + l'estimation)
#   bash scripts/run_all_ragas.sh --yes      # lance sans confirmation (ex. nohup)
#   REPEATS=2 bash scripts/run_all_ragas.sh  # réduire le nombre de répétitions

set -uo pipefail   # PAS -e : on continue même si un run échoue (timeout RAGAS ponctuel)

# Racine du repo (le script est dans scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

# On garantit des runs COMPLETS sur le jeu figé (pas de limite, pas de dataset alternatif).
unset RAGAS_LIMIT_QUESTIONS RAGAS_DATASET_PATH

# ---------- Paramètres éditables ----------
REPEATS=${REPEATS:-5}                                 # nb de runs par condition (variance)
RUN_EXTRA=${RUN_EXTRA:-1}                              # 1 = passe métriques extra par condition clé
EXTRA_METRICS="answer_correctness,aspect_critic"
EXTRA_SUFFIX="_with_correctness_aspect"                # suffixe produit par evaluate_ragas pour ces extra
EST_MIN_PER_RUN=${EST_MIN_PER_RUN:-18}                # estimation minutes / run (4 métriques)
EST_MIN_PER_EXTRA=${EST_MIN_PER_EXTRA:-25}            # estimation minutes / run (6 métriques)

# Conditions de variance : "label_humain|SQL_GENERATION_MODE|HYBRID_MODE"
CONDITIONS=(
  "controlled_sql|controlled|sql_only"
  "controlled_hybrid|controlled|sql_with_rag_context"
  "llm_sql|llm|sql_only"
)
# Conditions pour la passe métriques extra (1× chacune) : mêmes champs
EXTRA_CONDITIONS=(
  "controlled_sql|controlled|sql_only"
  "llm_sql|llm|sql_only"
)

RESULTS_DIR="evaluation/results"
LOG_DIR="$RESULTS_DIR/logs"
ARCHIVE_DIR="$RESULTS_DIR/variance_runs"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/run_all_${STAMP}.log"

log() { echo "$@" | tee -a "$MASTER_LOG"; }
fmt_min() { printf "%dh%02dm" $(( $1 / 60 )) $(( $1 % 60 )); }

# ---------- Construire la liste des runs ----------
# Chaque entrée : "kind|sgm|hm|idx|est_min"
RUNS=()
for c in "${CONDITIONS[@]}"; do
  IFS='|' read -r _ sgm hm <<< "$c"
  for i in $(seq 1 "$REPEATS"); do
    RUNS+=("variance|$sgm|$hm|$i|$EST_MIN_PER_RUN")
  done
done
if [ "$RUN_EXTRA" = "1" ]; then
  for c in "${EXTRA_CONDITIONS[@]}"; do
    IFS='|' read -r _ sgm hm <<< "$c"
    RUNS+=("extra|$sgm|$hm|1|$EST_MIN_PER_EXTRA")
  done
fi

TOTAL=${#RUNS[@]}
EST_TOTAL=0
for r in "${RUNS[@]}"; do IFS='|' read -r _ _ _ _ est <<< "$r"; EST_TOTAL=$((EST_TOTAL + est)); done

# ---------- Plan + confirmation ----------
log "=== Plan RAGAS — $STAMP ==="
log "Runs : $TOTAL   (variance ${REPEATS}× sur ${#CONDITIONS[@]} conditions + $([ "$RUN_EXTRA" = 1 ] && echo "${#EXTRA_CONDITIONS[@]} passes extra" || echo "0 passe extra"))"
log "Durée totale estimée : ~$(fmt_min "$EST_TOTAL")   (≈ ${EST_TOTAL} min)"
log "Coût API approximatif : ~$(( (TOTAL * 200) )) appels mistral-large/embeddings (ordre de grandeur)."
log "Logs : $LOG_DIR    | archives par run : $ARCHIVE_DIR"
n=0
for r in "${RUNS[@]}"; do
  IFS='|' read -r kind sgm hm idx est <<< "$r"; n=$((n + 1))
  log "  [$n/$TOTAL] $kind  SQL=$sgm HYBRID=$hm  run=$idx  (~${est} min)"
done
if [ "${1:-}" != "--yes" ]; then
  echo
  read -r -p "Lancer ces $TOTAL runs (~$(fmt_min "$EST_TOTAL")) ? [Entrée = oui, Ctrl-C = annuler] " _ || exit 1
fi

# ---------- Boucle d'exécution ----------
START=$SECONDS
remaining_est=$EST_TOTAL
done_runs=0
declare -a SUMMARY_LINES=()

for r in "${RUNS[@]}"; do
  IFS='|' read -r kind sgm hm idx est <<< "$r"
  done_runs=$((done_runs + 1))
  elapsed_min=$(( (SECONDS - START) / 60 ))

  label="routed_${sgm}_${hm}"
  [ "$kind" = "extra" ] && label="${label}${EXTRA_SUFFIX}"
  canon_sum="$RESULTS_DIR/ragas_${label}_summary.json"
  canon_csv="$RESULTS_DIR/ragas_${label}_results.csv"
  arch_sum="$ARCHIVE_DIR/ragas_${label}_run${idx}_summary.json"
  arch_csv="$ARCHIVE_DIR/ragas_${label}_run${idx}_results.csv"
  run_log="$LOG_DIR/${label}_run${idx}_${STAMP}.log"

  log ""
  log ">>> [$done_runs/$TOTAL] $kind | SQL=$sgm HYBRID=$hm run=$idx | écoulé $(fmt_min "$elapsed_min") | restant estimé ~$(fmt_min "$remaining_est")"
  log "    log : $run_log"

  marker=$(mktemp)   # pour détecter un fichier de résultat fraîchement écrit
  extra_env=""
  [ "$kind" = "extra" ] && extra_env="RAGAS_EXTRA_METRICS=$EXTRA_METRICS"

  # Run (env appliqué uniquement à cette commande). On NE coupe pas le script si ça échoue.
  env SQL_GENERATION_MODE="$sgm" HYBRID_MODE="$hm" $extra_env \
    poetry run python scripts/evaluate_ragas.py --eval-mode routed 2>&1 | tee "$run_log"

  # Bilan du run : complet ? + archivage (uniquement si le résumé a bien été (ré)écrit).
  status="ÉCHEC (pas de résumé)"
  if [ -f "$canon_sum" ] && [ "$canon_sum" -nt "$marker" ]; then
    cp "$canon_sum" "$arch_sum"; cp "$canon_csv" "$arch_csv"
    if grep -q "Évaluation complète : OUI" "$run_log"; then status="OK (complet)"; else status="INCOMPLET (scores manquants)"; fi
  fi
  rm -f "$marker"
  SUMMARY_LINES+=("[$done_runs/$TOTAL] $label run$idx -> $status")
  log "    => $status | archivé : $arch_sum"

  remaining_est=$((remaining_est - est))
done

TOTAL_MIN=$(( (SECONDS - START) / 60 ))
log ""
log "=== Terminé en $(fmt_min "$TOTAL_MIN") ==="
for line in "${SUMMARY_LINES[@]}"; do log "  $line"; done

# Comparaison finale (sans API) sur les derniers fichiers canoniques de chaque condition.
log ""
log "=== Comparaison (derniers runs canoniques) ==="
poetry run python scripts/compare_ragas_runs.py 2>&1 | tee -a "$MASTER_LOG"

log ""
log "Tous les runs sont archivés dans : $ARCHIVE_DIR (1 fichier par répétition)."
log "Pour moyenner la variance, traiter les fichiers $ARCHIVE_DIR/ragas_routed_<condition>_run*_summary.json."
log "Log maître : $MASTER_LOG"
