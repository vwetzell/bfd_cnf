#!/bin/bash
# HISTORICAL: this ran on 2026-08-13 and the `_mcb` flows it produced ARE the
# canonical flows/{bulk,shear}{,_gauss2}.eqx now; the pre-ceiling control it
# refers to below was moved to flows/pre_mcbound/.  Kept as the record of how
# the current flows were made -- to rebuild them, use retrain_all.sh.
#
# Retrain bulk+shear for both populations under the slot-2 (Mc/Mr) point-source
# ceiling added to `RawMomentStandardize`.  The `_mcb` flows are the experiment;
# the existing flows/{bulk,shear}{,_gauss2}.eqx from retrain_all.sh are the
# control, trained on identical data at the identical settings under the old
# bare-ratio chart.
#
# gauss2 first: it is where the support-leak mechanism was actually confirmed
# (`dev/check_support_leak.py --exact-box`), and where the predicted effect is
# largest -- m1 collapses to -0.41 at its edge, far above the ~0.0008 run-to-run
# rebuild noise ([[chain-rebuild-noise-floor]]), so a null there is meaningful.
cd "$(dirname "$0")" || exit 1
D=../bfd_cnf_imsims/data
set -x
python bulk.py  train --data $D/gauss2_g0_1M.fits --flow flows/bulk_gauss2_mcb.eqx  || exit 1
python shear.py train --data $D/gauss2_g0_1M.fits --flow flows/shear_gauss2_mcb.eqx --init flows/bulk_gauss2_mcb.eqx || exit 1
python bulk.py  train --data $D/moments.fits      --flow flows/bulk_mcb.eqx         || exit 1
python shear.py train --data $D/moments.fits      --flow flows/shear_mcb.eqx        --init flows/bulk_mcb.eqx        || exit 1
echo "MCBOUND RETRAIN DONE"
