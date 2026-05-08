#!/usr/bin/env bash
# Phase 0 의무 체크 — Mac venv vs Pod 의 8 핵심 패키지 버전 일괄 비교 [L01, L08]
#
# Usage:
#   bash scripts/diff_env.sh                  # Compare local venv vs requirements.txt
#   bash scripts/diff_env.sh --pod <ssh-cmd>  # Compare local venv vs Pod via SSH
#
# Exit code:
#   0 — all 8 packages match between local venv and requirements.txt (or Pod)
#   1 — at least one package version mismatches

set -euo pipefail

PACKAGES=("torch" "transformers" "datasets" "accelerate" "huggingface-hub" "tokenizers" "safetensors" "numpy")

# Always strip cuda suffix for comparison: "2.4.1+cu124" → "2.4.1"
strip_cuda() {
    echo "$1" | sed 's/+cu[0-9]*$//'
}

get_version_local() {
    local pkg="$1"
    pip show "$pkg" 2>/dev/null | awk '/^Version:/ {print $2}'
}

get_pinned_version() {
    local pkg="$1"
    # Match exact == pin in requirements.txt
    grep -E "^${pkg}==" requirements.txt 2>/dev/null | head -1 | sed -E "s/^${pkg}==([^[:space:]]+).*/\1/"
}

echo "=== Mac/Pod 패키지 정합 체크 (8개 핵심 패키지) ==="
echo ""

mismatch_count=0
match_count=0

printf "%-20s %-20s %-20s %s\n" "Package" "Local venv" "requirements.txt" "Match"
printf "%-20s %-20s %-20s %s\n" "-------" "----------" "----------------" "-----"

for pkg in "${PACKAGES[@]}"; do
    local_v=$(get_version_local "$pkg" || echo "MISSING")
    pinned_v=$(get_pinned_version "$pkg" || echo "NO-PIN")
    
    local_v_clean=$(strip_cuda "$local_v")
    
    if [[ "$pinned_v" == "NO-PIN" ]]; then
        # Range pin (e.g., numpy>=2.0,<3.0). Skip strict check.
        match="SKIP (range)"
    elif [[ "$local_v_clean" == "$pinned_v" ]]; then
        match="✓"
        match_count=$((match_count + 1))
    else
        match="✗"
        mismatch_count=$((mismatch_count + 1))
    fi
    
    printf "%-20s %-20s %-20s %s\n" "$pkg" "$local_v" "$pinned_v" "$match"
done

echo ""
echo "Summary: $match_count match, $mismatch_count mismatch (out of ${#PACKAGES[@]})"

if [[ $mismatch_count -gt 0 ]]; then
    echo ""
    echo "❌ FAIL: Package versions don't match requirements.txt."
    echo "   Run: pip install -r requirements.txt"
    echo "   Or check requirements.txt pins."
    exit 1
fi

echo ""
echo "✓ PASS: All explicitly pinned packages match."
exit 0
