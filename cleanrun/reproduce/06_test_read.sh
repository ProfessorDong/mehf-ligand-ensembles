#!/bin/bash
# THE TEST READ. Gated on purpose. ~1.5 h. GPU required.
#
# This split has already been read FOUR times. Every read so far was governed by a
# pre-registration written and hashed beforehand. If you run this, you are spending
# another read: decide that deliberately, not by running the next script in a list.
#
# Set I_AM_SPENDING_A_TEST_READ=yes to proceed.
source "$(dirname "$0")/_common.sh"; preflight
cd "$CLEANRUN/code" || die "cannot cd to code/"
if [ "${I_AM_SPENDING_A_TEST_READ:-no}" != "yes" ]; then
  echo "Refusing. Read the note at the top of this script, then re-run with:"
  echo "  I_AM_SPENDING_A_TEST_READ=yes $0"
  exit 1
fi
CKPT=${CKPT:-$CLEANRUN/checkpoints}
SSTAR=${SSTAR:-$CLEANRUN/frozen/s_star_frozen_v2.json}
say "s* comes from $SSTAR (frozen before the read); analysis script hash:"
sha256sum "$CLEANRUN/code/scripts/mehf_internal_geometry_spectrum.py"
echo "  recorded: 023815a7895958a43783cd76d1b6d9013007e45b0bced6505df3418268c76dca"
mkdir -p "$OUT/test_read"
bash "$CLEANRUN/code/scripts/clean_protocol_test_lane.sh" clean 1
say "one session, read once. If a run dies before producing output, STOP and report it;"
say "a restart with the SAME command is permitted, a restart with changed settings is not."
