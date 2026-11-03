#!/usr/bin/env bash
# Wait for an idle GPU, then collect timing (bench.py) and hardware counters
# for the slides:  data/bench.txt, data/counters.txt
#   bash slides/measure.sh            (run from kernels/gemm)
set -u
cd "$(dirname "$0")/.."
DATA=slides/data
mkdir -p "$DATA"

pick_idle_gpu() {
    while true; do
        for g in $(rocm-smi --showuse 2>/dev/null | awk -F'[][]' '/GPU use/ {split($0,a,": "); if (a[3]+0 <= 3) print $2}'); do
            ok=1
            for _ in 1 2 3; do
                sleep 10
                u=$(rocm-smi -d "$g" --showuse 2>/dev/null | awk -F': ' '/GPU use/ {print $NF+0}')
                (( u > 3 )) && { ok=0; break; }
            done
            (( ok )) && { echo "$g"; return; }
        done
        sleep 30
    done
}

GPU=$(pick_idle_gpu)
echo "using GPU $GPU ($(date +%T))"
export HIP_VISIBLE_DEVICES=$GPU

python3 bench.py | tee "$DATA/bench.txt"

: > "$DATA/counters.raw"
for case in v0_128x128 v1 v2; do
    for grp in "SQ_LDS_BANK_CONFLICT SQ_INSTS_LDS" "TCC_HIT_sum TCC_MISS_sum" \
               "TCC_EA0_RDREQ_sum TCC_EA0_RDREQ_32B_sum" "SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE"; do
        out=/tmp/pmc_${case}_$RANDOM
        rocprofv3 --pmc $grp --kernel-include-regex "gemm_v" -d "$out" -o pmc -f csv \
            -- python3 bench.py "$case" > /dev/null 2>&1
        f=$(find "$out" -name "*counter_collection.csv" | head -1)
        [[ -n "$f" ]] && python3 - "$case" "$f" >> "$DATA/counters.raw" <<'EOF'
import csv, sys, collections
case, path = sys.argv[1], sys.argv[2]
per = collections.defaultdict(lambda: collections.defaultdict(float))
for r in csv.DictReader(open(path)):
    per[r["Dispatch_Id"]][r["Counter_Name"]] += float(r["Counter_Value"])
names = sorted({n for d in per.values() for n in d})
for n in names:
    vals = [d[n] for d in per.values() if n in d]
    print(case, n, sum(vals) / len(vals))
EOF
    done
done

python3 - "$DATA/counters.raw" > "$DATA/counters.txt" <<'EOF'
import sys, collections
c = collections.defaultdict(dict)
for line in open(sys.argv[1]):
    case, n, v = line.split()
    c[case][n] = float(v)
print("# per dispatch: LDS bank conflict cycles, L2 hit rate, HBM/MALL read GB, MFMA busy (% of CU-cycles)")
for case, d in c.items():
    hit = d.get("TCC_HIT_sum", 0) / max(1, d.get("TCC_HIT_sum", 0) + d.get("TCC_MISS_sum", 0))
    rd = (d.get("TCC_EA0_RDREQ_sum", 0) - d.get("TCC_EA0_RDREQ_32B_sum", 0)) * 64 + d.get("TCC_EA0_RDREQ_32B_sum", 0) * 32
    busy = d.get("SQ_VALU_MFMA_BUSY_CYCLES", 0) / max(1, d.get("GRBM_GUI_ACTIVE", 0) * 304)
    print(case, f"bank_conflict={d.get('SQ_LDS_BANK_CONFLICT', 0):.0f}",
          f"L2_hit={hit*100:.1f}%", f"EA_read={rd/1e9:.2f}GB", f"MFMA_busy={busy*100:.1f}%")
EOF
cat "$DATA/counters.txt"
echo "done $(date +%T)"
