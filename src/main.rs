use std::io::Error;
use std::fs;
use clap::{arg, Command};

use longest_common_substring::run;

fn main() -> Result<(), Error> {
    let matches = Command::new("Longest Common Substring")
        .version("0.1.0")
        .author("Nate Armstrong <naterarmstrong@gmail.com>")
        .about("Finds the longest common bytestring present in at least MIN_MATCHES of the input files.")
        .arg(
            arg!(-k --min_matches "The minimum number of files the bytestring must be present within.").required(false).default_value("2"),
        )
        .arg(
            arg!(<files> "The paths to the input files.").required(true).min_values(2),
        )
        .get_matches();

    let k: u32 = matches.value_of("min_matches").map(|k| k.parse()).unwrap_or(Ok(2)).unwrap();
    let files: Vec<&str> = matches.values_of("files").unwrap().collect();
    //println!("{}", k);
    //println!("{:?}", files);

    let mut input: Vec<Vec<u8>> = Vec::new();
    for filename in &files {
        input.push(fs::read(filename)?);
    }

    let out = run(&input, k);
    println!("Found the longest common substring to be of length {} at positions:", out.0);
    for (file_idx, offset) in out.1 {
        println!("    Offset {} in file {}.", offset, files[file_idx]);
    }

    Ok(())
}