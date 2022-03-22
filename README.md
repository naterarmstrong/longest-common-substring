# Longest Common Substring
This project is to learn rust by implementing a solution to the longest common substring problem of finding the longest common bytestring among `N` different files, which is present in at least `K` of them.

The implementation scales linearly both with the combined length of the files, as well as the number of files in which the eventual result must be present.
The implementation is based upon the paper by Babenko and Starikovskaya[^fn1], but does introduce some novel ideas by only keeping track of sentinel positions, and not imposing that sentinels are unique letters in the alphabet.
By making this change, we maintain the ability to store the bytes as a `u8`, and reuse an existing data structure used in the paper to do so.

### Use
Clone the repo, and run `cargo build --release` to build at `target/release/longest-common-substring`.
Assuming it is aliased as `lcs`, It is used on the command line as `lcs [-k num_docs_lcs_is_in] file1 file2 ...`.

The repo also has a bash script to run local benchmarks while varying the number of files, which can be used as `./run_hyperfine_benchmark` after installing [hyperfine](https://github.com/sharkdp/hyperfine).
This will print the time taken for each file count from 2 to 10 to the command line, and also write it to a markdown file at `benchmark_results/results.md`.

### Benchmark
| Number of files | Mean [ms] | Min [ms] | Max [ms] |
|:---|---:|---:|---:|
| 2 | 5.1 ± 1.5 | 3.8 | 13.4 |
| 3 | 9.1 ± 3.0 | 6.2 | 18.5 |
| 4 | 10.1 ± 1.7 | 7.9 | 14.8 |
| 5 | 12.6 ± 2.3 | 9.0 | 17.9 |
| 6 | 11.7 ± 1.0 | 10.4 | 14.4 |
| 7 | 13.4 ± 1.2 | 11.5 | 19.9 |
| 8 | 14.0 ± 0.9 | 12.6 | 17.5 |
| 9 | 14.3 ± 0.7 | 13.3 | 17.0 |
| 10 | 15.1 ± 0.8 | 14.1 | 21.3 |

#### Reference
[^fn1]: Babenko M.A., Starikovskaya T.A. (2008) Computing Longest Common Substrings Via Suffix Arrays. In: Hirsch E.A., Razborov A.A., Semenov A., Slissenko A. (eds) Computer Science – Theory and Applications. CSR 2008. Lecture Notes in Computer Science, vol 5010. Springer, Berlin, Heidelberg