#!/bin/bash
# ============================================================
#  Sector Research Orchestrator
#
#  Runs all 11 sector expert agents (sequentially to respect
#  API rate limits), then the adversarial judge reviews all.
#
#  Usage:
#    bash scripts/run_sector_research.sh              # all sectors
#    bash scripts/run_sector_research.sh technology    # one sector
#    bash scripts/run_sector_research.sh --top 20      # top 20 per sector
# ============================================================

set -uo pipefail

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"

PYTHON="${PYTHON:-/usr/local/bin/python3.12}"
TOP="${TOP:-100}"
SECTORS=(
    technology
    healthcare
    financials
    consumer_discretionary
    consumer_staples
    energy
    industrials
    materials
    real_estate
    utilities
    communication_services
)

# Parse args
SINGLE_SECTOR=""
for arg in "$@"; do
    case "$arg" in
        --top)  shift; TOP="$1"; shift ;;
        --top=*) TOP="${arg#*=}" ;;
        *)      SINGLE_SECTOR="$arg" ;;
    esac
done

echo "============================================================"
echo "  SECTOR RESEARCH — $(date)"
echo "  Top ${TOP} companies per sector"
echo "============================================================"

START_TIME=$(date +%s)

if [ -n "$SINGLE_SECTOR" ]; then
    echo "  Running single sector: $SINGLE_SECTOR"
    $PYTHON -m agents.sectors.sector_expert --sector "$SINGLE_SECTOR" --top "$TOP"
else
    COMPLETED=0
    FAILED=0
    for sector in "${SECTORS[@]}"; do
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "  [$((COMPLETED + FAILED + 1))/11] Starting: $sector"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

        if $PYTHON -m agents.sectors.sector_expert --sector "$sector" --top "$TOP"; then
            COMPLETED=$((COMPLETED + 1))
        else
            FAILED=$((FAILED + 1))
            echo "  ⚠ $sector FAILED"
        fi
    done

    echo ""
    echo "============================================================"
    echo "  SECTOR ANALYSIS COMPLETE: $COMPLETED succeeded, $FAILED failed"
    echo "============================================================"
fi

# Run the adversarial judge
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ADVERSARIAL JUDGE — reviewing all sector research"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -n "$SINGLE_SECTOR" ]; then
    $PYTHON -m agents.sectors.sector_judge --sector "$SINGLE_SECTOR"
else
    $PYTHON -m agents.sectors.sector_judge
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
echo "============================================================"
echo "  TOTAL TIME: ${MINUTES}m ${SECONDS}s"
echo "  Reports: data/sector_research/"
echo "============================================================"
