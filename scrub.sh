#!/usr/bin/env bash
# usage: bash scrub.sh file1 file2 ...   (in-place; removes the internal project name)
for f in "$@"; do
  sed -i \
    -e 's/PhysioGuardCNN/BeatCNN/g' \
    -e 's/PhysioGuardTTA/GuardedTTA/g' \
    -e 's/physioguard\\\\_space/ttaguard/g' \
    -e 's#MyDrive/physioguard_space#MyDrive/ttaguard_data#g' \
    -e 's#MyDrive/physioguard#MyDrive/ttaguard_out#g' \
    -e 's#/content/physioguard_space#/content/ttaguard_repo#g' \
    -e "s/PhysioGuard-Space's/this project's/g" \
    -e 's/PhysioGuard-Space/this project/g' \
    -e "s/PhysioGuard's/our method's/g" \
    -e 's/PhysioGuard/our method/g' \
    -e 's/tab:physioguard_results/tab:ttaguard_results/g' \
    -e 's/physioguard_space/ttaguard/g' \
    -e 's/physioguard/ttaguard/g' \
    "$f"
done
