#!/bin/bash
# HANDOFF.md step 2, full chain: retrain.sh + gauss2 bulk/shear + centroid.
cd "$(dirname "$0")" || exit 1
set -x
bash retrain.sh || exit 1
python bulk.py  train --data ../bfd_cnf_imsims/data/gauss2_g0_1M.fits --flow flows/bulk_gauss2.eqx || exit 1
python shear.py train --data ../bfd_cnf_imsims/data/gauss2_g0_1M.fits --flow flows/shear_gauss2.eqx --init flows/bulk_gauss2.eqx || exit 1
python centroid.py train --copies ../bfd_cnf_imsims/data/copies_bulgedisc.fits --init flows/shear.eqx --flow flows/centroid.eqx || exit 1
echo "ALL RETRAIN DONE"
