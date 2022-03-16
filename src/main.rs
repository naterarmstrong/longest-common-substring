use suffix::SuffixTable;


fn main() {

    let s1 = "abcdef".to_string();
    let s2 = "xyzabcdfe".to_string();
    let sentinel = '!';

    let combined = format!("{}{}{}", s1, sentinel, s2);

    let st = SuffixTable::new(&combined);

    // maybe implement manber-myers algorithm?
    let lcp = get_lcp_array(&combined, st.table());
    let max_and_index = lcp.iter().enumerate().max_by(|(_, x), (_, y)| x.cmp(y)).unwrap();
    let start_index = st.table()[max_and_index.0] as usize;

    println!("{}", &combined);
    println!("{:?}", st);
    println!("{:?}", st.table());
    println!("{:?}", &lcp);
    println!("{:?}", max_and_index);
    println!("{}", &combined.as_str()[start_index..start_index + max_and_index.1]);
}


// Kasai construction algo
fn get_lcp_array(original_text: &String, suffix_array: &[u32]) -> Vec<usize> {
    let n = suffix_array.len();
    let s = original_text.as_bytes();
    let mut sa_inv: Vec<usize> = vec![0; n];

    for i in 0..n {
        sa_inv[suffix_array[i] as usize] = i;
    }

    let mut lcp: Vec<usize> = vec![0; n];

    let mut k = 0;

    for i in 0..n {
        if sa_inv[i] == n - 1 {
            k = 0;
            continue;
        }

        // Next substring
        let j: usize = suffix_array[sa_inv[i] + 1] as usize;

        while (i + k < n) && (j + k < n) && s[i + k] == s[j + k] {
            k += 1;
        }

        lcp[sa_inv[i]] = k;

        if k > 0 {
            k -= 1;
        }
    }

    return lcp;
}
