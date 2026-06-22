#!/usr/bin/env bash
# scripts/ocr/run_ragas.sh — Comparaison RAGAS selon le moteur OCR.
#
# But : mesurer l'impact du moteur OCR (captures Reddit) sur le RAG, proprement, en
# REPEATS runs par condition (défaut 5), sans rien écraser.
#
# Conditions (config V4 inchangée : SQL_GENERATION_MODE=llm, HYBRID_MODE=sql_only) :
#   - noocr    : index vector_db_noocr/    (texte natif seul, Reddit ignoré)   -> suffixe _noocr
#   - easyocr  : index vector_db_easyocr/  (OCR EasyOCR, moteur historique)     -> suffixe _easyocr
#   - nanonets : index vector_db_nanonets/ (OCR Nanonets-OCR-s, moteur amélioré)-> suffixe _nanonets
#
# NOTE : la condition « EasyOCR » correspond aussi aux 5 runs V4 de référence
# (`ragas_routed_llm_sql_only_run*`, index vector_db/ construit avec EasyOCR). Pour
# économiser, on peut RÉUTILISER ces runs comme condition EasyOCR et ne (re)lancer que
# Nanonets : ONLY=nanonets bash scripts/ocr/run_ragas.sh.
#
# Chaque run est archivé dans evaluation/results/variance_runs/ :
#   ragas_routed_llm_sql_only_{noocr,easyocr,nanonets}_run<idx>_{summary.json,results.csv}
#
# Pré-requis : l'index de la condition doit exister (sinon le construire en 2 temps,
# car OCR/PyTorch et faiss ne cohabitent pas sur Mac Apple Silicon) :
#   poetry run python scripts/ocr/extract_documents.py --ocr-engine nanonets --output vector_db_nanonets/documents.pkl
#   VECTOR_DB_DIR=vector_db_nanonets poetry run python indexer.py --documents vector_db_nanonets/documents.pkl
#
# Usage :
#   ONLY=nanonets bash scripts/ocr/run_ragas.sh --yes   # 5 runs Nanonets (EasyOCR réutilisé via V4)
#   bash scripts/ocr/run_ragas.sh                       # toutes les conditions présentes
#   REPEATS=3 ONLY=nanonets bash scripts/ocr/run_ragas.sh

set -uo pipefail   # PAS -e : on continue même si un run échoue (timeout RAGAS ponctuel)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1

# Runs COMPLETS sur le jeu figé (E01–E15) : pas de limite, pas de dataset alternatif.
unset RAGAS_LIMIT_QUESTIONS RAGAS_DATASET_PATH

REPEATS=${REPEATS:-5}
EST_MIN_PER_RUN=${EST_MIN_PER_RUN:-18}

# Conditions : "suffixe|VECTOR_DB_DIR". On peut en sélectionner une via ONLY=<suffixe>.
ALL_CONDITIONS=(
  "noocr|vector_db_noocr"
  "easyocr|vector_db_easyocr"
  "nanonets|vector_db_nanonets"
)
ONLY=${ONLY:-}
CONDITIONS=()
for c in "${ALL_CONDITIONS[@]}"; do
  IFS='|' read -r suffix _ <<< "$c"
  if [ -z "$ONLY" ] || [ "$ONLY" = "$suffix" ]; then CONDITIONS+=("$c"); fi
done
[ ${#CONDITIONS[@]} -eq 0 ] && { echo "ONLY=$ONLY inconnu (attendu : noocr, easyocr ou nanonets)"; exit 1; }

RESULTS_DIR="evaluation/results"
LOG_DIR="$RESULTS_DIR/logs"
ARCHIVE_DIR="$RESULTS_DIR/variance_runs"
mkdir -p "$LOG_DIR" "$ARCHIVE_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
MASTER_LOG="$LOG_DIR/run_ocr_${STAMP}.log"

log() { echo "$@" | tee -a "$MASTER_LOG"; }
fmt_min() { printf "%dh%02dm" $(( $1 / 60 )) $(( $1 % 60 )); }

TOTAL=$(( REPEATS * ${#CONDITIONS[@]} ))
EST_TOTAL=$(( TOTAL * EST_MIN_PER_RUN ))

log "=== Plan RAGAS OCR — $STAMP ==="
log "Runs : $TOTAL   (${REPEATS}× sur ${#CONDITIONS[@]} conditions : noocr + ocr)"
log "Durée totale estimée : ~$(fmt_min "$EST_TOTAL")   (≈ ${EST_TOTAL} min)"
log "Config : SQL_GENERATION_MODE=llm | HYBRID_MODE=sql_only (V4 inchangée)"
log "Index  : vector_db_noocr (sans OCR) vs vector_db_ocr (avec OCR)"
log "Logs : $LOG_DIR   | archives : $ARCHIVE_DIR"

# Vérifie que les deux index existent avant de lancer.
for c in "${CONDITIONS[@]}"; do
  IFS='|' read -r _ vdir <<< "$c"
  if [ ! -f "$vdir/faiss_index.idx" ]; then
    log "ERREUR : index introuvable ($vdir/faiss_index.idx). Construisez-le d'abord (voir l'en-tête du script)."
    exit 1
  fi
done

if [ "${1:-}" != "--yes" ]; then
  echo
  read -r -p "Lancer ces $TOTAL runs (~$(fmt_min "$EST_TOTAL")) ? [Entrée = oui, Ctrl-C = annuler] " _ || exit 1
fi

START=$SECONDS
done_runs=0
declare -a SUMMARY_LINES=()

for idx in $(seq 1 "$REPEATS"); do
  for c in "${CONDITIONS[@]}"; do
    IFS='|' read -r suffix vdir <<< "$c"
    done_runs=$((done_runs + 1))
    elapsed_min=$(( (SECONDS - START) / 60 ))
    remaining_est=$(( (TOTAL - done_runs + 1) * EST_MIN_PER_RUN ))

    label="routed_llm_sql_only_${suffix}"
    canon_sum="$RESULTS_DIR/ragas_${label}_summary.json"
    canon_csv="$RESULTS_DIR/ragas_${label}_results.csv"
    arch_sum="$ARCHIVE_DIR/ragas_${label}_run${idx}_summary.json"
    arch_csv="$ARCHIVE_DIR/ragas_${label}_run${idx}_results.csv"
    run_log="$LOG_DIR/${label}_run${idx}_${STAMP}.log"

    log ""
    log ">>> [$done_runs/$TOTAL] $suffix run=$idx | écoulé $(fmt_min "$elapsed_min") | restant estimé ~$(fmt_min "$remaining_est")"
    log "    index : $vdir | log : $run_log"

    marker=$(mktemp)
    env VECTOR_DB_DIR="$vdir" RAGAS_RUN_LABEL_SUFFIX="$suffix" \
        SQL_GENERATION_MODE="llm" HYBRID_MODE="sql_only" \
      poetry run python scripts/evaluate_ragas.py --eval-mode routed 2>&1 | tee "$run_log"

    status="ÉCHEC (pas de résumé)"
    if [ -f "$canon_sum" ] && [ "$canon_sum" -nt "$marker" ]; then
      cp "$canon_sum" "$arch_sum"; cp "$canon_csv" "$arch_csv"
      if grep -q "Évaluation complète : OUI" "$run_log"; then status="OK (complet)"; else status="INCOMPLET (scores manquants)"; fi
    fi
    rm -f "$marker"
    SUMMARY_LINES+=("[$done_runs/$TOTAL] $label run$idx -> $status")
    log "    => $status | archivé : $arch_sum"
  done
done

TOTAL_MIN=$(( (SECONDS - START) / 60 ))
log ""
log "=== Terminé en $(fmt_min "$TOTAL_MIN") ==="
for line in "${SUMMARY_LINES[@]}"; do log "  $line"; done
log ""
log "Archives : $ARCHIVE_DIR/ragas_routed_llm_sql_only_{noocr,ocr}_run*_summary.json"
log "Analyse  : notebooks/nanonets_ocr_analysis.ipynb"
log "Log maître : $MASTER_LOG"
