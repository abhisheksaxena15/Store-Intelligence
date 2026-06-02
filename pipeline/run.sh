#!/usr/bin/env bash
# pipeline/run.sh
# ===============
# Processes all CCTV clips for all stores and feeds events into the API.
#
# Usage:
#   ./run.sh [--data-dir /path/to/dataset] [--api-url http://localhost:8000] [--device cpu]
#
# Prerequisites:
#   pip install -r ../requirements.txt
#   The API must be running: docker compose up -d api

set -euo pipefail

DATA_DIR="${DATA_DIR:-../data}"
API_URL="${API_URL:-http://localhost:8000}"
DEVICE="${DEVICE:-cpu}"
EVENTS_DIR="${EVENTS_DIR:-../data/events}"
LAYOUT="${DATA_DIR}/store_layout.json"
POS_CSV="${DATA_DIR}/pos_transactions.csv"
MODEL="yolov8n.pt"
STRIDE=2   # process every 2nd frame

mkdir -p "${EVENTS_DIR}"

echo "=========================================="
echo "Store Intelligence — Detection Pipeline"
echo "Data dir : ${DATA_DIR}"
echo "API URL  : ${API_URL}"
echo "Device   : ${DEVICE}"
echo "=========================================="

# Discover store directories inside data/clips/
for store_dir in "${DATA_DIR}"/clips/*/; do
    store_id=$(basename "${store_dir}")
    echo ""
    echo ">>> Processing store: ${store_id}"

    for clip in "${store_dir}"*.mp4; do
        [ -f "${clip}" ] || continue

        filename=$(basename "${clip}" .mp4)
        # Derive camera_id from filename: entry → CAM_ENTRY_01, floor → CAM_FLOOR_01, etc.
        case "${filename,,}" in
            *entry*)  camera_id="CAM_ENTRY_01" ;;
            *floor*)  camera_id="CAM_FLOOR_01" ;;
            *billing*) camera_id="CAM_BILLING_01" ;;
            *) camera_id="CAM_$(echo "${filename}" | tr '[:lower:]' '[:upper:]')_01" ;;
        esac

        output="${EVENTS_DIR}/${store_id}_${camera_id}.jsonl"

        echo "  Processing clip: ${clip}"
        echo "  Camera ID      : ${camera_id}"
        echo "  Output         : ${output}"

        python detect.py \
            --video "${clip}" \
            --store-id "${store_id}" \
            --camera-id "${camera_id}" \
            --layout "${LAYOUT}" \
            --output "${output}" \
            --pos-csv "${POS_CSV}" \
            --model "${MODEL}" \
            --stride "${STRIDE}" \
            --device "${DEVICE}"

        echo "  Emitted events → ${output}"

        # Ingest into API in batches of 500
        python ingest_events.py \
            --events "${output}" \
            --api-url "${API_URL}" \
            --batch-size 500
    done
done

echo ""
echo "=========================================="
echo "All clips processed."
echo "Check API: ${API_URL}/stores/<store_id>/metrics"
echo "=========================================="