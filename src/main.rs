use suffix::SuffixTable;
use std::collections::VecDeque;

const K: u32 = 2;

fn main() {

    let s1 = "banana".to_string();
    let s2 = "appledbanabo".to_string();
    let s3 = "xyz_ban".to_string();
    let lengths = [s1.len(), s2.len(), s3.len()];
    let sentinel = '!';

    let combined = format!("{}{}{}{}{}{}", s1, sentinel, s2, sentinel, s3, sentinel);
    let idx_to_document = get_idx_to_document_array(&lengths);

    let st = SuffixTable::new(&combined);
    let lcp = get_lcp_array_from_suffix_array(combined.as_str().as_bytes(), st.table());
    let segments = calculate_segments(idx_to_document, K, lengths.len(), st.table());
    let lcs_metadata = process_segments(&segments, &lcp);
    let lcs = get_string_from_lcs_metadata(lcs_metadata.0, lcs_metadata.1, st.table(), &combined);
    print_lcs_output(&lengths, lcs, &st);

    /*println!("{}", &combined);
    println!("{:?}", st);
    println!("{:?}", st.lcp_lens());
    println!("{:?}", lcp);
    println!("{:?}", &segments);
    println!("{:?}", lcs_metadata);
    println!("{}", lcs); */
}

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
        We are interseted in keeping track of the minimum of a contiguous sequence: Use https://people.cs.uct.ac.za/~ksmith/articles/sliding_window_minimum.html to do it amortized O(1) time per index
*/

/* Represents the range [start, end).*/
#[derive(Debug)]
struct Segment {
    start: usize,
    end: usize,
}

// Kasai construction algo. TODO: proper reference
fn get_lcp_array_from_suffix_array(s: &[u8], suffix_array: &[u32]) -> Vec<usize> {
    let n = suffix_array.len();

    // Construct the rank array. rank is the inverse of suffix array as a permutation of 1..n
    let rank = get_inverse_of_permutation(suffix_array);

    let mut lcp: Vec<usize> = vec![0; n];

    let mut lcp_len = 0;

    for i in 0..n {
        if rank[i] == n - 1 {
            lcp_len = 0;
            continue;
        }

        // Next substring
        let j = suffix_array[rank[i] + 1] as usize;

        while (i + lcp_len < n) && (j + lcp_len < n) && s[i + lcp_len] == s[j + lcp_len] {
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

/* Returns the segments which are candidates for containing the longest common substring. */
fn calculate_segments(idx_to_document: Vec<usize>, k_requirement: u32, n_documents: usize, suffix_array: &[u32]) -> Vec<Segment> {
    let n = suffix_array.len();

    let mut k_count = 0;
    let mut buckets = vec![0; n_documents];
    let mut segments = Vec::new();
    let mut right_bound = n_documents;

    for i in n_documents..n {
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
            while !window.is_empty() && window.back().unwrap().val >= lcp_val {
                window.pop_back();
            }
            window.push_back(ValAndIdx {val: lcp_val, idx: window_end});
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

fn get_string_from_lcs_metadata<'a>(segment: &Segment, length: usize, suffix_array: &[u32], combined: &'a String) -> &'a str {
    let idx = suffix_array[segment.start] as usize;

    return &combined.as_str()[idx..idx + length];
}

fn print_lcs_output(document_lengths: &[usize], lcs: &str, st: &SuffixTable) {
    println!("Found the longest common substring to be of length {}", lcs.len());

    let locations = st.positions(lcs);
    for location in locations {
        let (doc_index, start_index) = get_document_index(document_lengths, *location as usize);
        println!("Found in document {} at index {}", doc_index, start_index);
    }
}

/* Returns (document_index, start_index). */
fn get_document_index(document_lengths: &[usize], mut index: usize) -> (usize, usize) {
    let mut i = 0;
    while index > document_lengths[i] {
        index -= document_lengths[i] + 1;
        i += 1;
    }
    return (i, index);
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
