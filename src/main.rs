use suffix::SuffixTable;


fn main() {

    let s1 = "abcdef".to_string();
    let s2 = "xyzabcdfe".to_string();
    let lengths = [s1.len(), s2.len()];
    let sentinel = '0';

    let combined = format!("{}{}{}{}", s1, sentinel, s2, sentinel);

    let st = SuffixTable::new(&combined);
    let lcp = get_lcp_array_from_suffix_array(combined.as_str().as_bytes(), st.table());
    let segments = calculate_segments(&lengths, 2, st.table());

    println!("{}", &combined);
    println!("{:?}", st);
    println!("{:?}", &segments);
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

/* Returns the segments which are candidates for containing the longest common substring. */
fn calculate_segments(input_lengths: &[usize], k_requirement: u32, suffix_array: &[u32]) -> Vec<Segment> {
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

    let mut k_count = 0;
    let mut buckets = vec![0; input_ct];
    let mut segments = Vec::new();
    let mut right_bound = input_ct;

    for i in input_ct..n {
        // Expand this segment until it is K-good
        while k_count < k_requirement && right_bound < n {
            let t_val = t[suffix_array[right_bound] as usize];
            if t_val != 0 {
                let bucket = t_val - 1;
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
        let t_val = t[suffix_array[i] as usize];
        if t_val != 0 {
            let bucket = t_val - 1;
            buckets[bucket] -= 1;
            if buckets[bucket] == 0 {
                k_count -= 1;
            }
        }

    }

    return segments;
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
