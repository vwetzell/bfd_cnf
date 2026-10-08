#!/bin/bash
# 2026-10-06 -- "all the complications" run (user): DES depth, noisy templates (bdg2d), noisy-template
# copies (tn), --deconvolve-templates, per-target noise +/-12% AND DES-like PSF variation measured from
# the Y6 grid (psf_hsm_moments: sigma +/-5% -> sd 0.020 around the sim's 0.40; e1/e2 sd 0.008, mean
# (+0.0008, -0.0039)).  Templates keep the fixed round deep-field PSF.
#   precision goal (user): sigma(m1) ~ 5e-4 combined = galaxy bootstrap ~4.2e-4 (1M/arm vs 5.6e-4 at
#   517k) + selection MC from 4 selection seeds at 2^24 (mean of 4 ~ 2^26; their spread IS the MC bar).
#   flow variance (user): seed-only, 2 extra independent chains (s3, s4) on the same templates/copies,
#   on the first 300k catalog rows (prefix of the main run -> paired with s2 there).
# Sigma_X coverage (pre-flight 2026-10-06): 1.34% of targets fell outside the 1.4 margin -> copies rebuilt at
#   --sigma-range 1.5 (0.40% outside) and every centroid stage (s2 too) retrained with --scale-margin 1.5.
# 2026-10-07: margin-1.5 centroid layers were a seed lottery (e-response 0.88-1.21) -- ONE noise-dominated
#   template (row 172291, copy-mean Mr = -17) set the z-space loss scale 10-1000x too large.  Copies culled by
#   dev/cull_copies.py (user rule: template dropped if >= 1% of copies have Mf<0, Mr<0 or |e|>1, otherwise
#   only those copies; Mc and the point-source bound not cut) -> tn15c, 136 templates dropped.  Two seeds on
#   tn15c at --n-aniso 400: gate 0.983/0.985, M1/M2 resid 2.1e-3, Sigma_X scan flat -> dev/eresp_scan.py is
#   now a gate too.  s2's centroid layer is the seed-2 test layer (identical command).  Bulk/shear keep
#   the full template set; only the centroid stage's copies are culled.
# Gates: pre-flight Sigma_X coverage / PD / PSF-spread (dev/desv_check.py) before any full render;
# tail gate + centroid e-response (0.97-1.03) on every new chain.  CPU renders, then GPU, strictly serial.
# Usage: bash dev/desv.sh > logs/desv/chain.log 2>&1
set -o pipefail
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; mkdir -p $L pqr
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; SM=1.5; C0=$D/copies_${POP}_bdg2d_tn15.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits
TPOP=bulgedisc_g2n_1m_desv; SX=$D/targets_bdg2_g0_1m_desv.fits
OBS="--noise-sigma 2.289 --psf-e1 0.0008 --psf-e2 -0.0039 --psf-sigma 0.40 --obs-sd 0.008 0.008 0.020 0.28"
BF="--pop $TPOP --samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates"
die() { echo "$1 FAILED $(date)"; exit 1; }
render() {   # out args...: skip if present, write via _tmp so a killed render never looks done
    local out=$1; shift
    [ -f $D/$out ] && return 0
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.$* --out data/${out%.fits}_tmp.fits \
        && mv data/${out%.fits}_tmp.fits data/$out ) 2>&1 | grep -v Warning
}

# --- CPU: noisy-template copies at the wider Sigma_X range (needed before the pre-flight reads its base) ---
render copies_${POP}_bdg2d_tn15.fits copies build --n 500000 --seed 0 --pop $POP --noise-sigma 2.289 \
    --template-noise 0.611 0.055 --sigma-range $SM --procs 24 || die "copies render"
rm -fv $D/copies_${POP}_bdg2d_tn.fits    # superseded 1.4-range set (user: regenerate if needed)
[ -f $C ] || { python -u dev/cull_copies.py $C0 ${C%.fits}_tmp.fits 0.01 > $L/cull_tn15.log 2>&1 \
    && mv ${C%.fits}_tmp.fits $C; } || die "copies cull"

# --- CPU: pre-flight on a 60k test render, then the 1M x 3 targets (seed 3) ---
render targets_bdg2_g0_60k_desv_test.fits sim --n 60000 --seed 103 --pop $POP $OBS --add-noise --nproc 24 \
    || die "test render"
python dev/desv_check.py $D/targets_bdg2_g0_60k_desv_test.fits $C $M $SM | tee $L/preflight.log
grep -q "PASS" $L/preflight.log || die "pre-flight"
python dev/des_match.py $D/targets_bdg2_g0_60k_desv_test.fits > $L/des_match.log 2>&1; cat $L/des_match.log
for arm in "g1p02 0.02" "g1m02 -0.02" "g0 0.0"; do
    set -- $arm
    render targets_bdg2_${1}_1m_desv.fits sim --n 1000000 --seed 3 --pop $POP $OBS --add-noise --g1 $2 \
        --nproc 24 || die "target render $1"
done
python dev/desv_check.py $SX $C $M $SM | tee $L/fullcheck.log; grep -q "PASS" $L/fullcheck.log || die "full-catalog check"
echo "renders done $(date)"

# --- GPU: centroid stage at the 1.5 margin for s2 (bulk/shear reused), then two more independent chains ---
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
for s in 2 3 4; do
    r=s$s
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || die bulk_base_$r
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || die bulk_$r
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || die shear_$r
    cf=$F/centroid_tnc15c_$r.eqx; cl=$L/centroid_tnc15c_$r.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; die "centroid_tnc15c_$r $1"; }
    [ -f $cf ] || systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M \
        --init $F/shear_$r.eqx --flow $cf --converge --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || cfail train
    gate centroid $cf || cfail "tail gate"
    er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || cfail "e-response gate (ratio $er)"
    python -u dev/eresp_scan.py $C $M $cf 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || cfail "Sigma_X scan gate"
    tail -1 $L/eresp_scan_c_$r.log
    echo "$r trained (e-response $er) $(date)"
done

# --- GPU: main run, flow s2, full 1M catalog ---
resolve() {   # tag pqr flow size_lo size_hi flux_lo flux_hi seed [extra]: selection-only re-solve
    local tag=$1 p=$2 fl=$3; shift 3
    [ -f $L/bias_$tag.log ] && grep -q "windowed, corrected" $L/bias_$tag.log && return 0
    python -u bias.py $BF --flow $fl --window-size $1 $2 --window-flux $3 $4 --seed $5 ${@:6} --load-pqr $p \
        > $L/bias_$tag.log 2>&1 || die "re-solve $tag"
    grep -E "P_s =|R_s =|windowed, corrected" $L/bias_$tag.log | sed "s/^/  $tag /"
}
W="full90k:2.2 3.5 2200 90000 sel20k:2.2 3.5 2200 20000 cv6win:2.2 3.2 3000 20000 s32_f2200:2.2 3.2 2200 20000 s35_f3000:2.2 3.5 3000 20000"
P2=pqr/desv_tnc15c_s2.npz
t0=$SECONDS
[ -f $P2 ] || python -u bias.py $BF --flow $F/centroid_tnc15c_s2.eqx --window-size 2.2 3.5 --window-flux 2200 90000 \
    --save-pqr $P2 > $L/bias_s2_main.log 2>&1 || die "main bias"
echo "WALL $((SECONDS - t0)) s" >> $L/bias_s2_main.log
grep -E "windowed|P_s =|Q_s =|R_s =|top draw|WALL|ESS =" $L/bias_s2_main.log
echo "$W" | tr ' ' '\n' | paste -d' ' - - - - | while read -r spec; do
    name=${spec%%:*}; set -- ${spec#*:}
    for seed in 0 1 2 3; do resolve s2_${name}_sd$seed $P2 $F/centroid_tnc15c_s2.eqx $1 $2 $3 $4 $seed; done
done
for h in 0.0025 0.01; do resolve s2_sel20k_h$h $P2 $F/centroid_tnc15c_s2.eqx 2.2 3.5 2200 20000 0 --window-fd $h; done

# --- GPU: flow variance, flows s2/s3/s4 on the first 300k catalog rows ---
# 2026-10-08: the first pass had N_ns from the FULL catalog's NPOP against a 300k slice (bias.py fixed:
#   NPOP now scaled to the slice) and re-solves without --n-targets; those logs are in $L/nns_bug/.
#   s2 needs no new integration: its pre-filtered rows for the prefix are the first rows of $P2 (checked
#   equal to s3/s4's, row for row), so the sliced copy pairs with s3/s4 exactly.
NT=300000
[ -f pqr/desv_tnc15c_s2_sub.npz ] || python - <<EOF2 || die "s2 prefix slice"
import numpy as np
a = np.load("$P2"); b = np.load("pqr/desv_tnc15c_s3_sub.npz"); n = len(b["moments"])
assert all(np.array_equal(a[k][:n], b[k]) for k in ("moments", "obs_plus", "obs_minus", "prefilter_w"))
np.savez_compressed("pqr/desv_tnc15c_s2_sub.npz", **{k: a[k][:n] for k in a.files}, target_rows=np.array([0, $NT]))
EOF2
for s in 3 4; do
    P=pqr/desv_tnc15c_s${s}_sub.npz
    [ -f $P ] || python -u bias.py $BF --flow $F/centroid_tnc15c_s$s.eqx --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets $NT --save-pqr $P > $L/bias_s${s}_sub.log 2>&1 || die "bias s$s subset"
done
for s in 2 3 4; do
    echo "$W" | tr ' ' '\n' | paste -d' ' - - - - | head -3 | while read -r spec; do
        name=${spec%%:*}; set -- ${spec#*:}
        for seed in 0 1 2 3; do resolve s${s}sub_${name}_sd$seed pqr/desv_tnc15c_s${s}_sub.npz $F/centroid_tnc15c_s$s.eqx \
            $1 $2 $3 $4 $seed --n-targets $NT; done
    done
done
echo "ALL DONE $(date)"
