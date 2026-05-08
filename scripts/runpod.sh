#!/usr/bin/env bash
# DEPRECATED — vast.ai 단일 사용 정책 (L37, 2026-05-08).
# RunPod 사용 금지. 본 스크립트는 historical reference 로 보존.
# 신규 phase 의 Pod 부팅은 CLAUDE.md §3 (vast.ai setup 7단계) + `vastai create instance` 직접 사용.
#
# Pod lifecycle management — Runpod CLI ≥1.14 신 인터페이스 [L03, L23]
#
# Usage:
#   bash scripts/runpod.sh up [--auto-recreate]    # Create new pod
#   bash scripts/runpod.sh start [--auto-recreate] # Start existing (auto-recreate on failure)
#   bash scripts/runpod.sh stop                    # Stop pod (preserve volume)
#   bash scripts/runpod.sh down                    # Terminate pod
#   bash scripts/runpod.sh status                  # Print pod state
#   bash scripts/runpod.sh ssh                     # SSH into pod
#
# Env vars:
#   RUNPOD_NETWORK_VOLUME_ID — Network volume ID (required)
#   RUNPOD_GPU_FALLBACK      — Space-separated GPU types (17 default)
#   RUNPOD_GPU_FALLBACK_LARGE— 80GB-class GPUs only (Phase 7 70B)
#   RUNPOD_IMAGE             — runpod/pytorch:2.4.0-py3.11-cuda12.4.1
#   RUNPOD_LARGE_GPU         — true to use FALLBACK_LARGE list

set -euo pipefail

VOLUME_ID="${RUNPOD_NETWORK_VOLUME_ID:-tf2ln0ukj3}"
IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1}"

# GPU fallback list (comma-separated, v3 style). 17 GPU types default.
DEFAULT_GPUS="NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB,NVIDIA RTX A6000,NVIDIA RTX A5000,NVIDIA RTX A4500,NVIDIA RTX A4000,NVIDIA L40S,NVIDIA L40,NVIDIA L4,NVIDIA A40,NVIDIA GeForce RTX 4090,NVIDIA GeForce RTX 3090,NVIDIA RTX 5090,NVIDIA RTX 4000 Ada Generation,NVIDIA RTX 2000 Ada Generation"
GPU_FALLBACK="${RUNPOD_GPU_FALLBACK:-$DEFAULT_GPUS}"

# 80GB+ only (Llama-70B 8-bit needs ~70GB)
DEFAULT_LARGE="NVIDIA H100 PCIe,NVIDIA H100 80GB HBM3,NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB"
GPU_FALLBACK_LARGE="${RUNPOD_GPU_FALLBACK_LARGE:-$DEFAULT_LARGE}"

POD_NAME="${RUNPOD_POD_NAME:-cacheblend-v4}"
SSH_KEY="${HOME}/.runpod/ssh/RunPod-Key-Go"

# ─── Helpers ────────────────────────────────────────────────────────────────

log() { echo "[runpod.sh] $*" >&2; }

get_pod_id() {
    runpodctl pod list --output json 2>/dev/null \
        | python3 -c "import json, sys
data = json.load(sys.stdin)
pods = data.get('pods', [])
for p in pods:
    if p.get('name') == '$POD_NAME':
        print(p.get('id', ''))
        sys.exit(0)
sys.exit(1)
" || echo ""
}

get_pod_json() {
    local pod_id="$1"
    runpodctl pod get "$pod_id" --output json 2>/dev/null
}

wait_for_ssh() {
    local pod_id="$1"
    local max_wait=180
    local waited=0
    log "Waiting for SSH endpoint (up to ${max_wait}s)..."
    while [[ $waited -lt $max_wait ]]; do
        local json
        json=$(get_pod_json "$pod_id")
        local ip
        ip=$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('ssh',{}).get('ip',''))" 2>/dev/null || echo "")
        if [[ -n "$ip" ]]; then
            log "SSH endpoint ready: $ip"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "ERROR: SSH endpoint not ready after ${max_wait}s"
    return 1
}

create_pod() {
    local gpu_list="$1"

    # Network volume DC must match Pod DC. Auto-resolve from volume metadata.
    local dc_id
    dc_id=$(runpodctl network-volume list 2>/dev/null \
        | python3 -c "import json,sys; d=json.load(sys.stdin); print(next((v['dataCenterId'] for v in d if v.get('id')=='$VOLUME_ID'), ''))" 2>/dev/null || echo "")
    if [[ -z "$dc_id" ]]; then
        log "ERROR: Could not resolve dataCenterId for network volume $VOLUME_ID"
        return 1
    fi
    log "Network volume $VOLUME_ID in dc=$dc_id; will pin Pod to same DC."

    # Comma-separated → IFS-aware iteration
    IFS=',' read -ra GPUS <<< "$gpu_list"
    for gpu in "${GPUS[@]}"; do
        # Trim whitespace
        gpu="${gpu#"${gpu%%[![:space:]]*}"}"
        gpu="${gpu%"${gpu##*[![:space:]]}"}"
        [[ -z "$gpu" ]] && continue

        log "Trying GPU type: $gpu"
        local out
        out=$(runpodctl pod create \
            --name "$POD_NAME" \
            --image "$IMAGE" \
            --gpu-id "$gpu" \
            --cloud-type SECURE \
            --network-volume-id "$VOLUME_ID" \
            --data-center-ids "$dc_id" \
            --container-disk-in-gb 30 \
            --output json 2>&1) || true
        
        local pod_id
        pod_id=$(echo "$out" | python3 -c "
import json, sys
try:
    d = json.loads(sys.stdin.read())
    print(d.get('id', ''))
except Exception:
    sys.exit(1)
" 2>/dev/null || echo "")
        
        if [[ -n "$pod_id" ]]; then
            log "OK Pod created: $pod_id (GPU: $gpu)"
            wait_for_ssh "$pod_id" || return 1
            echo "$pod_id"
            return 0
        else
            log "  Failed for $gpu, trying next..."
        fi
    done
    
    log "ERROR: All GPU types in fallback list failed"
    return 1
}

# ─── Commands ───────────────────────────────────────────────────────────────

cmd_up() {
    local auto_recreate=false
    [[ "${1:-}" == "--auto-recreate" ]] && auto_recreate=true
    
    local existing
    existing=$(get_pod_id)
    if [[ -n "$existing" ]]; then
        log "Pod already exists: $existing"
        echo "$existing"
        return 0
    fi
    
    local gpu_list="$GPU_FALLBACK"
    [[ "${RUNPOD_LARGE_GPU:-false}" == "true" ]] && gpu_list="$GPU_FALLBACK_LARGE"
    
    create_pod "$gpu_list"
}

cmd_start() {
    local auto_recreate=false
    [[ "${1:-}" == "--auto-recreate" ]] && auto_recreate=true
    
    local pod_id
    pod_id=$(get_pod_id)
    if [[ -z "$pod_id" ]]; then
        log "No existing pod, creating new..."
        cmd_up "$@"
        return $?
    fi
    
    log "Starting pod: $pod_id"
    if runpodctl pod start "$pod_id" 2>&1; then
        wait_for_ssh "$pod_id"
        echo "$pod_id"
        return 0
    fi
    
    if [[ "$auto_recreate" == "true" ]]; then
        log "Start failed. Auto-recreate: terminate + new pod"
        runpodctl pod remove "$pod_id" 2>/dev/null || true
        local gpu_list="$GPU_FALLBACK"
        [[ "${RUNPOD_LARGE_GPU:-false}" == "true" ]] && gpu_list="$GPU_FALLBACK_LARGE"
        create_pod "$gpu_list"
    else
        log "ERROR: Start failed. Use --auto-recreate to terminate + create new."
        return 1
    fi
}

cmd_stop() {
    local pod_id
    pod_id=$(get_pod_id)
    [[ -z "$pod_id" ]] && { log "No pod found"; return 1; }
    log "Stopping pod: $pod_id"
    runpodctl pod stop "$pod_id"
}

cmd_down() {
    local pod_id
    pod_id=$(get_pod_id)
    [[ -z "$pod_id" ]] && { log "No pod found"; return 1; }
    log "Terminating pod: $pod_id"
    runpodctl pod remove "$pod_id"
}

cmd_status() {
    local pod_id
    pod_id=$(get_pod_id)
    if [[ -z "$pod_id" ]]; then
        echo "No pod with name $POD_NAME"
        return 1
    fi
    get_pod_json "$pod_id" | python3 -m json.tool
}

cmd_ssh() {
    local pod_id
    pod_id=$(get_pod_id)
    [[ -z "$pod_id" ]] && { log "No pod found"; return 1; }
    
    local json ip port
    json=$(get_pod_json "$pod_id")
    ip=$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ssh']['ip'])")
    port=$(echo "$json" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['ssh']['port'])")
    
    log "Connecting to root@${ip}:${port}"
    ssh -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$port" "root@${ip}"
}

# ─── Dispatch ───────────────────────────────────────────────────────────────

cmd="${1:-}"
shift || true

case "$cmd" in
    up)     cmd_up "$@" ;;
    start)  cmd_start "$@" ;;
    stop)   cmd_stop ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    ssh)    cmd_ssh ;;
    *)      echo "Usage: $0 {up|start|stop|down|status|ssh} [--auto-recreate]" >&2
            exit 1 ;;
esac
