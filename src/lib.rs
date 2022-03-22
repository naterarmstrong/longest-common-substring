use std::collections::VecDeque;
use suffix_array::SuffixArray;

const LCP_SENTINEL: usize = u32::MAX as usize;
const SENTINEL: u8 = 0;
const SENTINEL_IDX: usize = 0;

/*
Summary of info from `Computing Longest Common Substrings Via Suffix Arrays` by Babenko and Starikovskaya

Definitions:
    The rank is the inverse of the suffix array permutation of 1..n
    K-good means a set of positions that contains positions with from K different initial strings
    lcp(i) is the length of the longest common prefix of the ith and i+1th suffix (ordered in suffix array)
    delta_i is the shortest K-good segment starting at position i
    L_0 refers to the largest integer i with a K-good segment starting at i.
    w(i) is the minimum lcp(j) with j contained in the bounds of delta_i

Statements:
    For any pair of indices, the longest common prefix of their positions in the suffix array is the minimum
        of the pairwise lcp values between i and j.

Algorithm stages:
    Combine input strings into output of length L separated by sentinel values which are distinct and strictly
        smaller than all other letters in the alphabet.
    Construct suffix array and lcp array according to above definitions.
    When constructing lcp array, take advantage of the fact that lcp[rank(i+1]) >= lcp(rank[i]) - 1 to do it in O(n) time.

    Compute minimal segments starting at each index.. This can be done by stepping the right index until k-good,
     then initializing next right index at that same spot.

    To test for K-goodness quickly:
        Keep track of t(j) array which you index into to get the type of an index in the string.
        Keep a counter per document and a counter of how many are nonzero, update appropriately.

    To process the segments:
        We are interested in keeping track of the minimum of a contiguous sequence.
        Use an ordered deque to get each minimum in amortized O(1).
        Keep track of the segment and maximum length seen while iterating through.
    
    To get output:
        Every index found in the segment contains the LCP. Convert them back to document:offset, then return.
*/

/*
Modifications to only store the data as a byte array instead of going up to u16 array:
    Big Idea:
        Instead of letting the sentinels be distinct from the alphabet in question (which would then require going from u8 to u16),
        we keep track of the locations of the sentinels when constructing LCP array and calculating/processing segments.
        This lets us continue to use a library to construct the suffix array.
        If the suffix array construction was done manually, the rest of these changes would be less necessary.

    Use byte suffix array, use 0 byte as sentinel.
        Problem: There may be non-sentinel entries at the start of the array
        Solution: Skip past entries that start with a sentinel in LCP to still match consecutive strings in 'true' suffix array (that which would be constructed w/o sentinels)

    LCP Creation:
        Do the same algorithm as before, but pretend that values that start with a sentinel are not there. Similarly, break matching when hitting a sentinel.
        Fill in sentinel values with u32.MAX. We won't get that large of files anyways. Adding option would add overhead, ruin the point of keping it as u8 instead of u16.
        Result: LCP array which is correct, some values marked LCP_SENTINEL to be ignored.

    Calculate Segments:
        Once again, similar algo to before, but need to start from 0, and skip sentinels as we go. Algorithm is almost identical.

    Process Segments:
        Just increment past sentinels when iterating. Otherwise, keep the same algo

    Final segment:
        Every non-sentinel contained in this segment MUST have the LCS starting at the start index. So there's no need to match again.
*/

#[derive(Debug, PartialEq)]
pub struct LCSOutput {
    pub length: usize,
    pub positions: Vec<FilePosition>,
}

#[derive(Debug, PartialEq)]
pub struct FilePosition {
    pub file_index: usize,
    pub offset: usize,
}

/* Return LCSOutput representing the length and position of the longest common substring which exists in k or more of the input files. */
pub fn run(inputs: &Vec<Vec<u8>>, k: u32) -> LCSOutput {
    let lengths: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
    let total_size: usize = &lengths.iter().sum() + inputs.len();
    let mut combined = Vec::with_capacity(total_size);
    for input in inputs {
        combined.extend(input);
        combined.push(SENTINEL);
    }
    let idx_to_document = get_idx_to_document_array(&lengths);

    let sa = SuffixArray::new(&combined);
    let sa_raw = &sa.into_parts().1[1..];

    let rank = get_inverse_of_permutation(sa_raw);
    let lcp = get_lcp_array_from_suffix_array(&combined, sa_raw, &idx_to_document, &rank);
    let segments = calculate_segments(&idx_to_document, k, inputs.len(), sa_raw);
    let segment_and_lcs_length = process_segments(&segments, &lcp);
    let lcs_positions =
        get_positions_from_segment(segment_and_lcs_length.0, &lengths, sa_raw, &idx_to_document);
    LCSOutput {
        length: segment_and_lcs_length.1,
        positions: lcs_positions,
    }
}

/* Represents the range [start, end) within the suffix array.*/
#[derive(Debug, PartialEq)]
struct Segment {
    start: usize,
    end: usize,
}

/* Kasai's suffix array construction algorithm, slightly modified to use sentinel positions instead of lexicographically distinct sentinels. */
fn get_lcp_array_from_suffix_array(
    s: &[u8],
    suffix_array: &[u32],
    idx_to_document: &Vec<usize>,
    rank: &Vec<usize>,
) -> Vec<usize> {
    let n = suffix_array.len();

    let mut lcp: Vec<usize> = vec![0; n];

    let mut lcp_len = 0;

    // i := position in combined string
    for i in 0..n {
        if idx_to_document[i] == SENTINEL_IDX {
            lcp[rank[i]] = LCP_SENTINEL;
            continue;
        } else if rank[i] == n - 1 {
            lcp_len = 0;
            continue;
        }

        // Next substring
        let mut d_to_next_non_sentinel = 1;
        let mut j = suffix_array[rank[i] + d_to_next_non_sentinel] as usize;
        while idx_to_document[j] == SENTINEL_IDX {
            d_to_next_non_sentinel += 1;
            if rank[i] + d_to_next_non_sentinel >= n - 1 {
                lcp_len = 0;
                j = 0;
                break;
            }
            j = suffix_array[rank[i] + d_to_next_non_sentinel] as usize;
        }

        while (i + lcp_len < n)
            && (j + lcp_len < n)
            && s[i + lcp_len] == s[j + lcp_len]
            && idx_to_document[i + lcp_len] != SENTINEL_IDX
            && idx_to_document[j + lcp_len] != SENTINEL_IDX
        {
            lcp_len += 1;
        }

        lcp[rank[i]] = lcp_len;

        // Remove the first element, as we will increment
        if lcp_len > 0 {
            lcp_len -= 1;
        }
    }

    lcp
}

/* Returns a vector v such that for an index i in the combined string,
v[i] := the document index, 1-indexed. 0 represents a sentinel. */
fn get_idx_to_document_array(input_lengths: &[usize]) -> Vec<usize> {
    if input_lengths.len() == 0 {
        return Vec::new();
    }

    // The combined byte string has length equal to the sum of inputs, plus the sentinels dividing strings
    let n = input_lengths.iter().sum::<usize>() + input_lengths.len();
    let input_ct = input_lengths.len();

    // t[i] := the 1-based numbering of the document it belongs to, or 0 if it is a sentinel
    let mut t = vec![0; n];
    let mut cur_input = 0;
    let mut input_remaining = input_lengths[0];
    for i in 0..n {
        if input_remaining > 0 {
            t[i] = cur_input + 1;
            input_remaining -= 1;
        } else {
            t[i] = 0;
            cur_input += 1;
            if cur_input < input_ct {
                input_remaining = input_lengths[cur_input];
            }
        }
    }

    t
}

/* Returns the segments which are candidates for containing the longest common substring. */
fn calculate_segments(
    idx_to_document: &Vec<usize>,
    k_requirement: u32,
    n_documents: usize,
    suffix_array: &[u32],
) -> Vec<Segment> {
    let n = suffix_array.len();

    let mut k_count = 0;
    let mut buckets = vec![0; n_documents];
    let mut segments = Vec::new();
    let mut right_bound = 0;

    for i in 0..n {
        // Expand this segment until it is K-good
        while k_count < k_requirement && right_bound < n {
            let doc_val = idx_to_document[suffix_array[right_bound] as usize];
            if doc_val != SENTINEL_IDX {
                let bucket = doc_val - 1;
                if buckets[bucket] == 0 {
                    k_count += 1;
                }
                buckets[bucket] += 1;
            }
            right_bound += 1;
        }

        if right_bound == n {
            break;
        }

        // Push it to the list of segments
        segments.push(Segment {
            start: i,
            end: right_bound,
        });

        // Remove the start index
        let doc_val = idx_to_document[suffix_array[i] as usize];
        if doc_val != SENTINEL_IDX {
            let bucket = doc_val - 1;
            buckets[bucket] -= 1;
            if buckets[bucket] == 0 {
                k_count -= 1;
            }
        }
    }

    segments
}

#[derive(Debug)]
struct ValAndIdx {
    val: usize,
    idx: usize,
}

/* Keep track of the minimum lcp value between each segment's start and end. Return the maximum of those. 
   We use a sorted sliding window which results in an overall O(n) implementation using a deque.
   Heavily inspired by the sliding window minimum implementation described here:
   https://people.cs.uct.ac.za/~ksmith/articles/sliding_window_minimum.html
*/
fn process_segments<'a>(
    segments: &'a Vec<Segment>,
    lcp_array: &Vec<usize>,
) -> (&'a Segment, usize) {
    let mut max = 0;
    let mut max_segment = &segments[0];

    let mut window: VecDeque<ValAndIdx> = VecDeque::new();
    let mut window_end = 0;

    for segment in segments {
        // Add elements to sliding window
        while segment.end > window_end + 1 {
            let lcp_val = lcp_array[window_end];
            if lcp_val != LCP_SENTINEL {
                while !window.is_empty() && window.back().unwrap().val >= lcp_val {
                    window.pop_back();
                }
                window.push_back(ValAndIdx {
                    val: lcp_val,
                    idx: window_end,
                });
            }
            window_end += 1;
        }

        if !window.is_empty() && segment.start > window.front().unwrap().idx {
            window.pop_front();
        }

        let window_min = window.front().unwrap().val;
        if window_min > max {
            max = window_min;
            max_segment = &segment;
        }
    }

    (max_segment, max)
}

/* Returns (document_index, start_index)[]. */
fn get_positions_from_segment(
    segment: &Segment,
    lengths: &[usize],
    suffix_array: &[u32],
    idx_to_document: &[usize],
) -> Vec<FilePosition> {
    let mut positions = Vec::new();
    for i in (segment.start..segment.end)
        .filter(|v| idx_to_document[suffix_array[*v] as usize] != SENTINEL_IDX)
    {
        positions.push(get_file_position(lengths, suffix_array[i] as usize));
    }
    positions.sort_by(|a, b| (a.file_index).cmp(&b.file_index));
    positions
}

/* Returns (document_index, start_index). */
fn get_file_position(document_lengths: &[usize], mut index: usize) -> FilePosition {
    let mut i = 0;
    while index > document_lengths[i] {
        index -= document_lengths[i] + 1;
        i += 1;
    }
    FilePosition {
        file_index: i,
        offset: index,
    }
}

/* Construct the inverse of a permutation on 0..n.  */
fn get_inverse_of_permutation(permutation: &[u32]) -> Vec<usize> {
    let n = permutation.len();
    let mut inverse = vec![0; n];
    for i in 0..n {
        inverse[permutation[i] as usize] = i;
    }
    inverse
}

/*** Tests for small-scale correctness ***/
#[cfg(test)]
mod tests {
    use super::*;
    use suffix_array::SuffixArray;

    /* A basic correctness test for constructing the LCP array from string inputs. */
    #[test]
    fn correct_lcp_from_strings() {
        let s1 = "banana".to_string();
        let s2 = "appledbanab".to_string();
        let lengths = [s1.len(), s2.len()];
        let sentinel = '!';

        let combined = format!("{}{}{}{}", s1, sentinel, s2, sentinel);
        let idx_to_document = get_idx_to_document_array(&lengths);

        let sa_vec = SuffixArray::new(combined.as_bytes()).into_parts().1;
        let sa_raw = &sa_vec[1..];

        let rank = get_inverse_of_permutation(sa_raw);
        let lcp = get_lcp_array_from_suffix_array(
            combined.as_str().as_bytes(),
            sa_raw,
            &idx_to_document,
            &rank,
        );
        let sent = LCP_SENTINEL;

        assert_eq!(
            lcp,
            vec![sent, sent, 1, 1, 3, 3, 1, 0, 1, 4, 0, 0, 0, 0, 2, 2, 0, 1, 0]
        );
    }

    /* Checking that LCP construction handles characters with the same value as the sentinel. */
    #[test]
    fn correctly_handles_lcp_sentinel_duplicates() {
        let mut s1: Vec<u8> = vec![0, 0, 98, 97, 110]; // \0\0ban
        let mut s2: Vec<u8> = vec![97, 0, 0, 98, 120, 121]; // a\0\0bxy
        let lengths = [s1.len(), s2.len()];
        let idx_to_document = get_idx_to_document_array(&lengths);

        // Store combined in s1. A bit hacky, but works
        s1.push(0); // Sentinel
        s2.push(0); // Sentinel
        s1.extend(&s2);

        let sa_vec = SuffixArray::new(&s1).into_parts().1;
        let sa_raw = &sa_vec[1..];
        let rank = get_inverse_of_permutation(sa_raw);

        let lcp = get_lcp_array_from_suffix_array(&s1, sa_raw, &idx_to_document, &rank);
        let sent = LCP_SENTINEL;

        assert_eq!(sa_raw, vec![12, 0, 7, 5, 1, 8, 6, 3, 2, 9, 4, 10, 11]);
        assert_eq!(lcp, vec![sent, 3, 1, sent, 2, 0, 1, 0, 1, 0, 0, 0, 0]); // Notice that the 3rd entry is matching lcp with the 5th and skipping the sentinel
    }

    /* Checking that it finds the correct location and length of LCS when characters have same val as sentinel. */
    #[test]
    fn correctly_finds_lcs_len_that_contains_sentinel() {
        let mut s1: Vec<u8> = vec![0, 0, 98, 97, 110]; // \0\0ban
        let mut s2: Vec<u8> = vec![97, 0, 0, 98, 120, 121]; // a\0\0bxy
        let lengths = [s1.len(), s2.len()];
        let idx_to_document = get_idx_to_document_array(&lengths);

        // Store combined in s1. A bit hacky, but works
        s1.push(0); // Sentinel
        s2.push(0); // Sentinel
        s1.extend(&s2);

        let sa_vec = SuffixArray::new(&s1).into_parts().1;
        let sa_raw = &sa_vec[1..];
        let rank = get_inverse_of_permutation(sa_raw);

        let lcp = get_lcp_array_from_suffix_array(&s1, sa_raw, &idx_to_document, &rank);
        let segments = calculate_segments(&idx_to_document, 2, lengths.len(), sa_raw);
        let lcs_metadata = process_segments(&segments, &lcp);
        assert_eq!(lcs_metadata.0, &Segment { start: 0, end: 3 });
        assert_eq!(lcs_metadata.1, 3);
    }

    /* If sentinels were incorrectly accounted for, this would give lcs of [0, 0, 98] between input0 and input1. */
    #[test]
    fn run_avoids_sentinels() {
        let mut input = Vec::new();
        input.push(vec![0, 0, 98, 98]);
        input.push(vec![0, 98, 97, 96, 92, 98]);
        input.push(vec![0, 98, 96]);
        let out = run(&input, 2);
        let expected = LCSOutput {
            length: 2,
            positions: vec![
                FilePosition {
                    file_index: 1,
                    offset: 0,
                },
                FilePosition {
                    file_index: 2,
                    offset: 0,
                },
            ],
        };
        assert_eq!(out, expected);
    }

    /* If sentinels were incorrectly accounted for, this would give lcs of [0, 0, 98] between input0 and input1. */
    #[test]
    fn run_varies_k_properly() {
        let mut input = Vec::new();
        input.push(vec![0, 0, 98, 98]);
        input.push(vec![0, 98, 97, 96, 92, 98]);
        input.push(vec![0, 98, 96]);
        let out = run(&input, 2);
        let expected = LCSOutput {
            length: 2,
            positions: vec![
                FilePosition {
                    file_index: 1,
                    offset: 0,
                },
                FilePosition {
                    file_index: 2,
                    offset: 0,
                },
            ],
        };
        assert_eq!(out, expected);
    }
}
