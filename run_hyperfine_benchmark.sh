#!/bin/bash

# This file assumes it is run from the root directory of this repo
set -e

cargo build --release

filepaths=()
for i in {1..10};
do
    filepaths+=("example_data/sample.${i}")
done

rm -rf benchmark_results
mkdir benchmark_results

cmds=()
for n in {1..9}; do
    files=""
    for i in `seq 0 $n`; do
        files+=" "
        files+=${filepaths[$i]}
    done
    cmds+=("target/release/longest-common-substring $files")
done

hyperfine --warmup 5 --export-markdown benchmark_results/results.md "${cmds[@]}"
