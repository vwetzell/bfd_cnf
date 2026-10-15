#!/bin/zsh
# 2026-10-14: item 2 only (x6r), resumed past bulk_base_x6r's tail-gate FAIL (user: option 1 -- only its chart constants
# feed bulk --chart-from; every later stage keeps its gate).  Copy of dev/desv_x6r_psfe.sh minus item 3.
#  3. PSF-anisotropy leakage, paired: the desv 1M targets re-rendered with mean PSF e1 +0.05 (seed 3, same
#     galaxies/noise/conditions row for row; the population draw is n-dependent, so the full 1M is rendered),
#     pre-flight dev/desv_check.py, then bias x4rA on the first 300k vs pqr/desv_tnc15c_x4rA.npz.
#  2. third flow x6r through the whole cut procedure (seed 6, as x5r in dev/desv_night.sh), option-A bias
#     on the 300k slice + selection seeds 1-3, sel20k/cv6win re-solves.
# Gate failures stop that item only (reported), never retrained around.
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; SM=1.5; CUT=(--template-cut 1200 1.5)
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits; SX=$D/targets_bdg2_g0_1m_desv.fits
OBS=(--noise-sigma 2.289 --psf-e1 0.0508 --psf-e2 -0.0039 --psf-sigma 0.40 --obs-sd 0.008 0.008 0.020 0.28)
until grep -q "SEEDS90K DONE" $L/full90k_seeds_A.log; do sleep 30; done
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
gate() { python -u dev/tail_gate.py --kind $1 $2 --data $M --sx-from $SX 2>&1 | grep "tail gate" | tee -a $L/tail_gate.log | grep -q PASS; }
fail() { echo "$1 FAILED $(date)"; return 1; }
BF=(--samples 8192 --alpha 0.5 --chunk 4096 --batch-budget 32768 --no-zero-arm
  --support --floor-eps 0 --prefilter-pad 0.3 --prefilter-sample 0 --g 0.02 --window-fd 0.005
  --window-jacobian full --window-taper 0.05 150 --deconvolve-templates --template-noise-g $D/noiseG_bulgedisc_g2_bdg2d.npz $CUT)
rep() { grep -E 'dropp|outlier|top draw|traceless|windowed, (un)?corrected' $1 | tr -s ' ' | tr '\n' '|'; }
render() {   # out args...: skip if present, write via _tmp so a killed render never looks done
    local out=$1; shift
    [ -f $D/$out ] && return 0
    ( cd $S && OMP_NUM_THREADS=1 JAX_PLATFORMS=cpu python -u -m imsims.$* --out data/${out%.fits}_tmp.fits \
        && mv data/${out%.fits}_tmp.fits data/$out ) 2>&1 | grep -v Warning
}
psfe() {
    local T=psfe1p05 P=pqr/desv_tnc15c_x4rA_psfe1p05.npz
    for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
        set -- ${=arm}
        render targets_bdg2_${1}_1m_desv_$T.fits sim --n 1000000 --seed 3 --pop $POP $OBS --add-noise --g1 $2 \
            --nproc 24 || { fail "psfe render $1"; return 1; }
        [ $1 = g0 ] && { python dev/desv_check.py $D/targets_bdg2_g0_1m_desv_$T.fits $C $M $SM > $L/preflight_$T.log 2>&1;
            cat $L/preflight_$T.log; grep -q PASS $L/preflight_$T.log || { fail "psfe pre-flight"; return 1; }; }
    done
    echo "psfe renders done $(date)"
    python -u bias.py --pop bulgedisc_g2n_1m_desv_$T $BF --flow $F/centroid_tnc15c_x4r.eqx --window-size 2.2 3.5 \
        --window-flux 2200 90000 --n-targets 300000 --save-pqr $P > $L/bias_x4rA_$T.log 2>&1 || { fail "psfe bias"; return 1; }
    echo "PSFE BIAS DONE: $(rep $L/bias_x4rA_$T.log) $(date)"
}
bulk_stages() {  # tag seed
    local r=$1 s=$2
    [ -f $F/bulk_base_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_base_$r.eqx --converge $CUT \
        --batch 16384 --lr 3e-3 --tail-weight 1.0 --ema 0.9995 --seed $s > $L/bulk_base_$r.log 2>&1 \
        && gate bulk $F/bulk_base_$r.eqx; } || { fail bulk_base_$r; return 1; }
    [ -f $F/bulk_$r.eqx ] || { python -u bulk.py train --data $M --flow $F/bulk_$r.eqx --chart-from $F/bulk_base_$r.eqx $CUT \
        --jac-weight 5.0 --tail-weight 1.0 --converge --batch 16384 --lr 3e-3 --ema 0.9995 --seed $s > $L/bulk_$r.log 2>&1 \
        && gate bulk $F/bulk_$r.eqx; } || { fail bulk_$r; return 1; }
    echo "$r bulk: $(grep -h 'CONVERGED\|kept rows' $L/bulk_base_$r.log $L/bulk_$r.log | tr '\n' ' ') $(date)"
}
shear_stage() {  # tag seed
    local r=$1 s=$2
    [ -f $F/shear_$r.eqx ] || { python -u shear.py train --data $M --flow $F/shear_$r.eqx --init $F/bulk_$r.eqx $CUT --jac-weight 5.0 \
        --deriv-weight 1e4 --converge --batch 16384 --lr 6e-3 --ema 0.9995 --seed $s > $L/shear_$r.log 2>&1 \
        && gate shear $F/shear_$r.eqx; } || { fail shear_$r; return 1; }
    echo "$r shear: $(grep -h 'CONVERGED\|^val nll' $L/shear_$r.log | tr '\n' ' ') $(date)"
}
cent() {   # tag seed
    local r=$1 s=$2 cf=$F/centroid_tnc15c_$1.eqx cl=$L/centroid_tnc15c_$1.log
    cfail() { mv -f $cf ${cf%.eqx}.failed 2>/dev/null; fail "centroid_tnc15c_$r $1"; }
    [ -f $cf ] && return 0
    systemd-run --user --scope -q -p MemoryMax=48G -p MemorySwapMax=0 env \
        XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 XLA_FLAGS=--xla_gpu_enable_command_buffer= python -u centroid.py train-sigmax \
        --multi-scale --anisotropic --n-aniso 400 --aniso-max 0.10 --scale-margin $SM --copies $C --train-data $M $CUT \
        --init $F/shear_$r.eqx --flow $cf --steps 200000 --batch 8192 --seed $s \
        --g-resp-weight 1.0 --g-resp-grid --adapter > $cl 2>&1 || { cfail train; return 1; }
    gate centroid $cf || { cfail "tail gate"; return 1; }
    local er=$(awk '/ellipticity response/ {print $NF}' $cl | tail -1)
    awk -v x="$er" 'BEGIN {exit !(x > 0.97 && x < 1.03)}' || { cfail "e-response gate (ratio $er)"; return 1; }
    python -u dev/eresp_scan.py $C $M $cf $CUT 2>&1 | grep -v -e Warning -e log_den > $L/eresp_scan_c_$r.log || { cfail "Sigma_X scan gate"; return 1; }
    echo "$r centroid: e-resp $er; $(tail -1 $L/eresp_scan_c_$r.log); last full-loss $(grep -o 'full-loss (all 403 scales) [0-9.]*' $cl | tail -1 | awk '{print $NF}')  $(date)"
}
biasrun() {  # tag
    local r=$1 cf=$F/centroid_tnc15c_$1.eqx P=pqr/desv_tnc15c_$1A.npz B=(--pop bulgedisc_g2n_1m_desv $BF)
    [ -f $P ] || python -u bias.py $B --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 \
        --n-targets 300000 --save-pqr $P > $L/bias_${r}A.log 2>&1 || { fail "bias $r"; return 1; }
    echo "$r bias full90k sd0: $(rep $L/bias_${r}A.log)"
    for sd in 1 2 3; do
        python -u bias.py $B --flow $cf --window-size 2.2 3.5 --window-flux 2200 90000 --seed $sd --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}A_full90k_sd$sd.log 2>&1 || fail "re-solve $r sd$sd"
        echo "  $r full90k sd$sd: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}A_full90k_sd$sd.log | tr -s ' ' | tr '\n' '|')"
    done
    for spec in "sel20k:2.2 3.5 2200 20000" "cv6win:2.2 3.2 3000 20000"; do
        local name=${spec%%:*}; set -- ${=${spec#*:}}
        python -u bias.py $B --flow $cf --window-size $1 $2 --window-flux $3 $4 --seed 0 --n-targets 300000 \
            --load-pqr $P > $L/bias_${r}A_${name}_sd0.log 2>&1 || fail "re-solve $r $name"
        echo "  $r $name: $(grep -E 'top draw|windowed, corrected' $L/bias_${r}A_${name}_sd0.log | tr -s ' ' | tr '\n' '|')"
    done
}
echo "=== 2: x6r $(date)"
{ bulk_stages x6r 6 && shear_stage x6r 6 && cent x6r 6 && biasrun x6r; } || echo "item 2 stopped"
echo "ALL DONE $(date)"
