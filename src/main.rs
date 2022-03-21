use longest_common_substring::run;

fn main() {
    let input = vec!["banana".as_bytes().to_owned(), "xbanax".as_bytes().to_owned(), "ybany".as_bytes().to_owned()];
    let out = run(&input, 2);
    println!("{:?}", out);
}