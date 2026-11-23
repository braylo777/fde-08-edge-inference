//! CLI: `fde-08-edge-inference <op> [file.json]` (reads JSON input from the file or stdin).
//! `fde-08-edge-inference ops` lists the operations.
use std::io::Read;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let op = args.get(1).map(String::as_str).unwrap_or("ops");
    if op == "gguf_file" {
        // Inspect a real .gguf on disk: reads only the first 64 MB (header + tensor table).
        use std::io::Read as _;
        let path = args.get(2).expect("usage: fde-08-edge-inference gguf_file <model.gguf>");
        let mut buf = Vec::new();
        std::fs::File::open(path).expect("open gguf").take(64 << 20).read_to_end(&mut buf).expect("read gguf");
        match fde_08_edge_inference::gguf::inspect(&buf) {
            Ok(mut r) => {
                r["metadata"] = serde_json::json!("(omitted; use gguf_inspect for the full map)");
                println!("{}", serde_json::to_string_pretty(&r).unwrap())
            }
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(1)
            }
        }
        return;
    }
    if op == "ops" || op == "--help" || op == "-h" {
        println!("fde-08-edge-inference: Plan, quantize and route LLM inference on customer hardware: GGUF inspection, int4 and AWQ quantization, memory fit, offline routing.");
        println!("usage: fde-08-edge-inference <op> [input.json]   (input JSON from file or stdin)");
        println!("       fde-08-edge-inference gguf_file <model.gguf>");
        for o in fde_08_edge_inference::OPS {
            println!("  {o}");
        }
        return;
    }
    let mut raw = String::new();
    match args.get(2) {
        Some(p) => raw = std::fs::read_to_string(p).expect("read input file"),
        None => {
            std::io::stdin().read_to_string(&mut raw).expect("read stdin");
        }
    }
    let input: serde_json::Value = serde_json::from_str(if raw.trim().is_empty() { "null" } else { &raw })
        .expect("input must be JSON");
    let req = serde_json::json!({"op": op, "input": input}).to_string();
    let out: serde_json::Value = serde_json::from_str(&fde_08_edge_inference::dispatch(&req)).unwrap();
    println!("{}", serde_json::to_string_pretty(&out).unwrap());
    if out["ok"] != true {
        std::process::exit(1);
    }
}
