#!/bin/bash
# Retrain every flow to CONVERGENCE.  Sequential: one GPU, and the shear flows
# warm-start from the bulk ones.
#
# The bulk step count is 150000, not the 4000 this used to run.  That is not a
# tuning knob -- 4000 leaves the bulk visibly unconverged, and the amount it is
# unconverged by silently sets the measured shear bias:
#
#   bulk steps   val nll   window mass vs catalog   noiseless m1
#         4000   40.4405                   +3.70%        -0.0126
#        20000   40.4199                   +1.70%        -0.0227
#        60000   40.4070                   +0.83%        -0.0428
#       150000         -                   +0.61%        -0.0600
#
# Both columns are monotone, so any intermediate step count is a choice about
# which error to hide.  Only the converged end is a property of the METHOD
# rather than of the schedule.  m1 = -0.060 there is the honest number, and it
# is reproducible across three independent shear trainings (6000/20000/60000
# steps give -0.064/-0.056/-0.067, i.e. flat -- the SHEAR layer converges at
# the default, only the bulk did not).
#
# "window mass" is the fraction of the prior inside the target selection window
# (bulk.SIZE_WINDOW/FLUX_WINDOW) after convolution with C_M, against the
# catalog's own -- `bias.window_prob`.  The catalog's own sampling error is
# +/-0.47%, so +0.61% is at the floor 100k templates can resolve.
#
# Cost: ~21 min per bulk on a 4080S, against ~40 s before.  The previous,
# undertrained flows are kept in flows/pre_convergence/.
cd "$(dirname "$0")" || exit 1
D=../bfd_cnf_imsims/data
BULK_STEPS=150000
set -x
python bulk.py  train --data $D/moments.fits        --flow flows/bulk.eqx        --steps $BULK_STEPS || exit 1
python bulk.py  train --data $D/moments_sersic.fits --flow flows/bulk_sersic.eqx --steps $BULK_STEPS || exit 1
python shear.py train --data $D/moments.fits        --flow flows/shear.eqx        --init flows/bulk.eqx        || exit 1
python shear.py train --data $D/moments_sersic.fits --flow flows/shear_sersic.eqx --init flows/bulk_sersic.eqx || exit 1
python shear.py train --data $D/moments_sersic.fits --flow flows/shear_sersic_long.eqx --init flows/bulk_sersic.eqx --steps 20000 || exit 1
python centroid.py train --copies $D/copies_bulgedisc.fits --flow flows/centroid.eqx --init flows/shear.eqx --steps 16000 || exit 1
echo "all flows retrained"
