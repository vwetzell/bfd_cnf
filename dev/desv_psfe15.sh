#!/bin/zsh
# 2026-10-15 (user: "Let's go with the e=0.15 test"): PSF-anisotropy leakage, paired, at mean PSF e1 +0.15
# (0.1508; sd and everything else unchanged): the desv 1M targets re-rendered (seed 3, same galaxies/noise/
# conditions row for row), pre-flight dev/desv_check.py, bias x4rA on the first 300k vs pqr/desv_tnc15c_x4rA.npz.
# Analysis: python dev/psfe_desv_paired.py pqr/desv_tnc15c_x4rA.npz logs/desv/bias_x4rA.log 1m_desv \
#           pqr/desv_tnc15c_x4rA_psfe1p15.npz logs/desv/bias_x4rA_psfe1p15.log 1m_desv_psfe1p15 0.15
cd "$(dirname "$0")/.."
export XLA_PYTHON_CLIENT_PREALLOCATE=false
S=../bfd_cnf_imsims; D=$S/data; L=logs/desv; F=flows/desm; SM=1.5; CUT=(--template-cut 1200 1.5)
POP=bulgedisc_g2; M=$D/moments_${POP}_bdg2d.fits; C=$D/copies_${POP}_bdg2d_tn15c.fits
OBS=(--noise-sigma 2.289 --psf-e1 0.1508 --psf-e2 -0.0039 --psf-sigma 0.40 --obs-sd 0.008 0.008 0.020 0.28)
T=psfe1p15; P=pqr/desv_tnc15c_x4rA_$T.npz
ollama_off() { for m in $(ollama ps 2>/dev/null | awk 'NR > 1 {print $1}'); do ollama stop $m >/dev/null 2>&1; done; }
( while kill -0 $$ 2>/dev/null; do ollama_off; sleep 30; done ) &
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
for arm in "g0 0.0" "g1p02 0.02" "g1m02 -0.02"; do
    set -- ${=arm}
    render targets_bdg2_${1}_1m_desv_$T.fits sim --n 1000000 --seed 3 --pop $POP $OBS --add-noise --g1 $2 --nproc 24 \
        || { echo "render $1 FAILED $(date)"; exit 1; }
    if [ $1 = g0 ]; then
        python dev/desv_check.py $D/targets_bdg2_g0_1m_desv_$T.fits $C $M $SM > $L/preflight_$T.log 2>&1
        cat $L/preflight_$T.log; grep -q PASS $L/preflight_$T.log || { echo "pre-flight FAILED $(date)"; exit 1; }
    fi
done
echo "renders done $(date)"
[ -f $P ] || python -u bias.py --pop bulgedisc_g2n_1m_desv_$T $BF --flow $F/centroid_tnc15c_x4r.eqx --window-size 2.2 3.5 \
    --window-flux 2200 90000 --n-targets 300000 --save-pqr $P > $L/bias_x4rA_$T.log 2>&1 || { echo "bias FAILED $(date)"; exit 1; }
echo "BIAS DONE: $(rep $L/bias_x4rA_$T.log) $(date)"
echo "ALL DONE $(date)"
