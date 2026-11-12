#!/usr/bin/env bash

# TODO Aweful as hell. Fix that.

lammps_file="$1"

# Find the line number where the atom data starts
HEADER_END_LINE=$(grep -n "Atoms" $lammps_file | cut -d: -f1)
head -n "$HEADER_END_LINE" $lammps_file > subsampled_$lammps_file
echo '' >> subsampled_$lammps_file
tail -n +$((HEADER_END_LINE + 1)) $lammps_file | \
    awk 'BEGIN{srand()} rand() < 0.01' | \
    awk '{ $1 = NR; print }' >> subsampled_$lammps_file
echo '' >> subsampled_$lammps_file

# Use 'sed' to replace the old atom count on line 3
NUM_ATOMS=$(awk -v h="$HEADER_END_LINE" 'END{print NR-h-2}' subsampled_$lammps_file)
sed -i "4s/.*/$NUM_ATOMS atoms/" subsampled_$lammps_file
