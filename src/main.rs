use clap::{arg, Command};
use std::fs;
use std::io::Error;

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

    let k: u32 = matches
        .value_of("min_matches")
        .map(|k| k.parse())
        .unwrap_or(Ok(2))
        .unwrap();
    let files: Vec<&str> = matches.values_of("files").unwrap().collect();

    if (k as usize) > files.len() {
        println!(
            "Cannot find a substring that exists in {} files when only {} files are provided.",
            k,
            files.len()
        );
        return Ok(());
    }

    let mut input: Vec<Vec<u8>> = Vec::new();
    for filename in &files {
        input.push(fs::read(filename)?);
    }

    let out = run(&input, k);
    println!(
        "Found the longest common substring to be of length {} at positions:",
        out.length
    );
    for position in out.positions {
        println!(
            "    Offset {} in file {}.",
            position.offset, files[position.file_index]
        );
    }

    Ok(())
}
