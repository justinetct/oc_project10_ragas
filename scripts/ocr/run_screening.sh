#!/usr/bin/env bash
# scripts/ocr/run_screening.sh — Screening RAGAS des variantes OCR (1 run par variante).
#
# Pour l'expérience d'optimisation : on ne lance PAS 5 runs par variante (trop long), mais
# 1 seul run de screening par variante, config V4 inchangée (llm / sql_only, --eval-mode routed).
# Le meilleur candidat sera ensuite relancé en 5 runs (scripts/ocr/run_ragas.sh adapté).
#
# Chaque variante : "suffixe|VECTOR_DB_DIR". Résultats archivés (sans écrasement) :
#   evaluation/results/variance_runs/ragas_routed_llm_sql_only_<suffixe>_run1_{summary.json,results.csv}
#
# Usage : bash scripts/ocr/run_screening.sh --yes

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.." || exit 1
unset RAGAS_LIMIT_QUESTIONS RAGAS_DATASET_PATH

# Variantes de screening (nettoyage + chunking renforcé sur les docs OCR/Reddit).
VARIANTS=(
  "nanonets_ocr_clean|vector_db_nanonets_clean"
  "nanonets_ocr_clean_chunk_800_overlap_150|vector_db_nanonets_clean_800_150"
  "nanonets_ocr_clean_chunk_1000_overlap_200|vector_db_nanonets_clean_1000_200"
  "nanonets_ocr_clean_chunk_1200_overlap_250|vector_db_nanonets_clean_1200_250"
)

RESULTS_DIR="evaluation/results"
ARCHIVE_DIR="$RESULTS_DIR/variance_runs"
LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$ARCHIVE_DIR" "$LOG_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)

[ "${1:-}" != "--yes" ] && { echo "Lance ${#VARIANTS[@]} runs de screening (~$(( ${#VARIANTS[@]} * 18 )) min). Relancer avec --yes."; exit 1; }

n=0
for v in "${VARIANTS[@]}"; do
  IFS='|' read -r suffix vdir <<< "$v"
  n=$((n + 1))
  label="routed_llm_sql_only_${suffix}"
  run_log="$LOG_DIR/${label}_run1_${STAMP}.log"
  echo ">>> [$n/${#VARIANTS[@]}] screening $suffix (index $vdir)"
  if [ ! -f "$vdir/faiss_index.idx" ]; then echo "   index absent ($vdir), variante ignorée."; continue; fi
  env VECTOR_DB_DIR="$vdir" RAGAS_RUN_LABEL_SUFFIX="$suffix" \
      SQL_GENERATION_MODE="llm" HYBRID_MODE="sql_only" \
    poetry run python scripts/evaluate_ragas.py --eval-mode routed > "$run_log" 2>&1
  canon_sum="$RESULTS_DIR/ragas_${label}_summary.json"
  canon_csv="$RESULTS_DIR/ragas_${label}_results.csv"
  if [ -f "$canon_sum" ]; then
    cp "$canon_sum" "$ARCHIVE_DIR/ragas_${label}_run1_summary.json"
    cp "$canon_csv" "$ARCHIVE_DIR/ragas_${label}_run1_results.csv"
    echo "   => OK, archivé : ragas_${label}_run1_summary.json"
  else
    echo "   => ÉCHEC (pas de résumé). Voir $run_log"
  fi
done
echo "=== Screening terminé ($STAMP) ==="
