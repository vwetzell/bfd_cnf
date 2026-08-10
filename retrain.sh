#!/bin/bash
# Retrain every flow on the catalogs regenerated after the bfd weight fix.
# Sequential: one GPU, and the shear flows warm-start from the bulk ones.
cd "$(dirname "$0")" || exit 1
D=../bfd_cnf_imsims/data
set -x
python bulk.py  train --data $D/moments.fits        --flow flows/bulk.eqx        || exit 1
python bulk.py  train --data $D/moments_sersic.fits --flow flows/bulk_sersic.eqx || exit 1
python shear.py train --data $D/moments.fits        --flow flows/shear.eqx        --init flows/bulk.eqx        || exit 1
python shear.py train --data $D/moments_sersic.fits --flow flows/shear_sersic.eqx --init flows/bulk_sersic.eqx || exit 1
python shear.py train --data $D/moments_sersic.fits --flow flows/shear_sersic_long.eqx --init flows/bulk_sersic.eqx --steps 20000 || exit 1
echo "all flows retrained"
