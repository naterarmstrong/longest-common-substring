use suffix_array::SuffixArray;
use std::{collections::{VecDeque, HashSet}};

const LCP_SENTINEL: u32 = u32::MAX;
const SENTINEL: u8 = 0;

/* 
Algorithm Definitions:
    The rank is the inverse of the suffix array permutation of 1..n
    K-good means a set of positions that contains positions with from K different initial strings
    lcp(i) is the length of the longest common prefix of the ith and i+1th suffix (ordered in suffix array)
    delta_i is the shortest K-good segment starting at position i
    L_0 refers to the largest integer i with a K-good segment starting at i. 
    w(i) is the minimum lcp(j) with j contained in the bounds of delta_i

Paper statements:
    For any pair of indices, the longest common prefix of their positions in the suffix array is the minimum
        of the pairwise lcp values between i and j.
    
    lcp(rank(i+1)) >= lcp(rank(i)) - 1 so we can skip checking the first lcp(rank(i)) -1 letters when computing
        the value for rank(i+1)

Algorithm stages:
    Combine input strings into output of length L, construct suffix array and lcp array

    Compute delta_{n+1} -> delta_{L_0}.
        This can be done by stepping the right index until k-good, then initializing next one at that same spot

    To test for K-goodness quickly:
        Keep track of t(j) array which you index into to get the type of an index in the string
        Keep a counter per document and a counter of how many are nonzero, update appropriately
    
    To maintain values from (2):
        We are interested in keeping track of the minimum of a contiguous sequence: Use https://people.cs.uct.ac.za/~ksmith/articles/sliding_window_minimum.html to do it amortized O(1) time per index
*/

/*
How to only store the data as a byte array instead of going up to u16 array:
    Use byte suffix array, use 0 byte as sentinel.
        Problem: There may be non-sentinel entries at the start of the array
        Solution: Skip past entries that start with a sentinel in LCP to still match consecutive strings in true suffix array (that which would be constructed w/o sentinels)
    
    LCP Creation:
        Do the same algorithm as before, but pretend that values that start with a sentinel are not there. Similarly, break computation when hitting a sentinel.
        Maybe fill in sentinel values with u32.MAX or something. We won't get that large of files anyways. Adding option would add overhead, ruin the point of keping it as u8 instead of u16
        Result: LCP array which is correct, some values marked as ignored (assuming we can't handle combined file size above 2**32 bytes anyways, we will never get lcs of that length)
    
    Calculate Segments:
        Once again, similar algo to before, but need to start from 0, and skip sentinels as we go. Algo becomes almost identical.
    
    Process Segments:
        Just increment past sentinels when iterating. Otherwise, keep the same algo
    


    Suffix Array: ith position is the lexicographic order of the suffix s[i..]
    LCP Array: ith position is the lcp between ith ordered suffix and i+1th ordered suffix (ignoring sentinels)

    Final segment:
        Every non-sentinel contained in this segment MUST have the LCS starting at the start index. So there's no need to match again.
*/


#[derive(Debug)]
pub struct LCSOutput {
    pub length: usize,
    pub positions: Vec<FilePosition>
}

#[derive(Debug)]
pub struct FilePosition {
    pub file_index: usize,
    pub offset: usize,
}

pub fn run(inputs: &Vec<Vec<u8>>, k: u32) -> LCSOutput {
    let lengths: Vec<usize> = inputs.iter().map(|input| input.len()).collect();
    let total_size: usize = &lengths.iter().sum() + inputs.len();
    let mut combined = Vec::with_capacity(total_size);
    for input in inputs {
        combined.extend(input);
        combined.push(SENTINEL);
    }
    let idx_to_document = get_idx_to_document_array(&lengths);
    let sentinel_pos = get_sentinel_checker(&lengths);

    let sa = SuffixArray::new(&combined);
    let sa_raw = &sa.into_parts().1[1..];
    
    let rank = get_inverse_of_permutation(sa_raw);
    let lcp = get_lcp_array_from_suffix_array(&combined, sa_raw, &idx_to_document, &rank);
    let segments = calculate_segments(&idx_to_document, k, inputs.len(), sa_raw);
    let lcs_metadata = process_segments(&segments, &lcp);
    let lcs_positions = get_positions_from_segment(lcs_metadata.0, &lengths, sa_raw, &idx_to_document);
    LCSOutput{ length: lcs_metadata.1, positions: lcs_positions}
}

/* Represents the range [start, end).*/
#[derive(Debug, PartialEq)]
struct Segment {
    start: usize,
    end: usize,
}

// Kasai construction algo. TODO: proper reference
fn get_lcp_array_from_suffix_array(s: &[u8], suffix_array: &[u32], idx_to_document: &Vec<usize>, rank: &Vec<usize>) -> Vec<usize> {
    let n = suffix_array.len();

    let mut lcp: Vec<usize> = vec![0; n];

    let mut lcp_len = 0;

    // i := position in combined string
    for i in 0..n {
        // Check this for sanity. Should I be checking if i is a sentinel, or if rank[i] is a sentinel...
        if idx_to_document[i] == 0 {
            lcp[rank[i]] = LCP_SENTINEL as usize;
            continue;
        } else if rank[i] == n - 1 {
            lcp_len = 0;
            continue;
        }

        // Next substring
        let mut d_to_next_non_sentinel = 1;
        let mut j = suffix_array[rank[i] + d_to_next_non_sentinel] as usize; //TODO: iterate until we hit a non-sentinel
        while idx_to_document[j] == 0 {
            d_to_next_non_sentinel += 1;
            if rank[i] + d_to_next_non_sentinel >= n - 1 {
                // TODO: handle this case properly
                lcp_len = 0;
                continue;
            }
            j = suffix_array[rank[i] + d_to_next_non_sentinel] as usize;
        }

        while (i + lcp_len < n) && (j + lcp_len < n) && s[i + lcp_len] == s[j + lcp_len] && idx_to_document[i + lcp_len] != 0 && idx_to_document[j + lcp_len] != 0 { //TODO: If either one is a sentinel, say they are not equal
            lcp_len += 1;
        }

        lcp[rank[i]] = lcp_len;

        // Remove the first element, as we will increment
        if lcp_len > 0 {
            lcp_len -= 1;
        }
    }

    return lcp;
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

    return t;
}

fn get_sentinel_checker(input_lengths: &[usize]) -> HashSet<usize> {
    let mut cur_pos = 0;
    let mut sentinel_set = HashSet::new();
    for length in input_lengths {
        cur_pos += length;
        sentinel_set.insert(cur_pos+1);
        cur_pos += 1;
    }
    sentinel_set
}

/* Returns the segments which are candidates for containing the longest common substring. */
fn calculate_segments(idx_to_document: &Vec<usize>, k_requirement: u32, n_documents: usize, suffix_array: &[u32]) -> Vec<Segment> {
    let n = suffix_array.len();

    let mut k_count = 0;
    let mut buckets = vec![0; n_documents];
    let mut segments = Vec::new();
    let mut right_bound = 0;

    for i in 0..n {
        // Expand this segment until it is K-good
        while k_count < k_requirement && right_bound < n {
            let doc_val = idx_to_document[suffix_array[right_bound] as usize];
            if doc_val != 0 {
                let bucket = doc_val - 1;
                // It's not a sentinel
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
        segments.push(Segment{start: i, end: right_bound});

        // Remove the start index
        let doc_val = idx_to_document[suffix_array[i] as usize];
        if doc_val != 0 {
            let bucket = doc_val - 1;
            buckets[bucket] -= 1;
            if buckets[bucket] == 0 {
                k_count -= 1;
            }
        }

    }

    return segments;
}

#[derive(Debug)]
struct ValAndIdx{
    val: usize, 
    idx: usize,
}

/* Keep track of the minimum lcp value between each segment's start and end. Return the maximum of those. */
fn process_segments<'a>(segments: &'a Vec<Segment>, lcp_array: &Vec<usize>) -> (&'a Segment, usize) {
    if segments.len() == 0 {
        panic!("No segments passed into process segments"); // TODO; use result
    }
    let mut max = 0;
    let mut max_segment = &segments[0];

    let mut window: VecDeque<ValAndIdx> = VecDeque::new();
    let mut window_end = 0;

    for segment in segments {
        // Add elements to sliding optimized window to contain 
        while segment.end > window_end + 1 {
            let lcp_val = lcp_array[window_end];
            if lcp_val != (LCP_SENTINEL as usize) {
                while !window.is_empty() && window.back().unwrap().val >= lcp_val {
                    window.pop_back();
                }
                window.push_back(ValAndIdx {val: lcp_val, idx: window_end});
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

    return (max_segment, max);
}

/* Returns (document_index, start_index)[]. */
fn get_positions_from_segment(segment: &Segment, lengths: &[usize], suffix_array: &[u32], idx_to_document: &[usize]) -> Vec<FilePosition> {
    let mut positions = Vec::new();
    for i in (segment.start..segment.end).filter(|v| idx_to_document[suffix_array[*v] as usize] != 0) {
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
    FilePosition{file_index: i, offset: index}
}

/* Construct the inverse of a permutation on 0..n.  */
fn get_inverse_of_permutation(permutation: &[u32]) -> Vec<usize> {
    let n = permutation.len();
    let mut inverse = vec![0; n];
    for i in 0..n {
        inverse[permutation[i] as usize] = i;
    }
    return inverse;
}




/*** Tests for small-scale correctness ***/
#[cfg(test)]
mod tests {
    use super::*;
    use suffix_array::SuffixArray;

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
        let lcp = get_lcp_array_from_suffix_array(combined.as_str().as_bytes(), sa_raw, &idx_to_document, &rank);
        let sent = LCP_SENTINEL as usize;

        assert_eq!(lcp, vec![sent, sent, 1, 1, 3, 3, 1, 0, 1, 4, 0, 0, 0, 0, 2, 2, 0, 1, 0]);
    }

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
        let sent = LCP_SENTINEL as usize;

        assert_eq!(sa_raw, vec![12, 0, 7, 5, 1, 8, 6, 3, 2, 9, 4, 10, 11]);
        assert_eq!(lcp, vec![sent, 3, 1, sent, 2, 0, 1, 0, 1, 0, 0, 0, 0]); // Notice that the 3rd entry is matching lcp with the 5th and skipping the sentinel
    }

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
        assert_eq!(lcs_metadata.0, &Segment {start: 0, end: 3});
        assert_eq!(lcs_metadata.1, 3);
    }

    #[test]
    fn run_parses_input_properly() {
        let mut input = Vec::new();
        input.push(vec![0, 98, 98]);
        input.push(vec![1, 3, 0, 98, 97]);
        input.push(vec![2, 97, 98]);
        println!("Run result: {:?}", run(&input, 2));
    }
}