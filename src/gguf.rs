//! GGUF (v2/v3) header reader. Reads only the header and tensor table, so a few MB from the
//! start of a multi-GB file are enough. All reads are bounds-checked; lengths are capped so a
//! hostile file cannot make the reader allocate gigabytes.

use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

struct R<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.i.checked_add(n).ok_or("length overflow")?;
        if end > self.b.len() {
            return Err(format!(
                "truncated at byte {} (need {n} more); pass more of the file",
                self.i
            ));
        }
        let s = &self.b[self.i..end];
        self.i = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > 1 << 20 {
            return Err(format!("string length {n} too large"));
        }
        Ok(String::from_utf8_lossy(self.take(n as usize)?).into_owned())
    }
    fn value(&mut self, t: u32, depth: usize) -> Result<Value, String> {
        Ok(match t {
            0 => json!(self.take(1)?[0]),
            1 => json!(self.take(1)?[0] as i8),
            2 => json!(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => json!(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => json!(self.u32()?),
            5 => json!(self.u32()? as i32),
            6 => json!(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            7 => json!(self.take(1)?[0] != 0),
            8 => json!(self.string()?),
            9 => {
                if depth > 2 {
                    return Err("nested arrays too deep".into());
                }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 24 {
                    return Err(format!("array length {n} too large"));
                }
                let mut items = Vec::new();
                for k in 0..n {
                    let v = self.value(et, depth + 1)?;
                    if k < 8 {
                        items.push(v); // keep a preview; vocab arrays have 100k+ entries
                    }
                }
                json!({"array_len": n, "preview": items})
            }
            10 => json!(self.u64()?),
            11 => json!(self.u64()? as i64),
            12 => json!(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            other => return Err(format!("unknown metadata value type {other}")),
        })
    }
}

/// (name, bytes per block, elements per block)
pub fn ggml_type(t: u32) -> (&'static str, u64, u64) {
    match t {
        0 => ("F32", 4, 1),
        1 => ("F16", 2, 1),
        2 => ("Q4_0", 18, 32),
        3 => ("Q4_1", 20, 32),
        6 => ("Q5_0", 22, 32),
        7 => ("Q5_1", 24, 32),
        8 => ("Q8_0", 34, 32),
        10 => ("Q2_K", 84, 256),
        11 => ("Q3_K", 110, 256),
        12 => ("Q4_K", 144, 256),
        13 => ("Q5_K", 176, 256),
        14 => ("Q6_K", 210, 256),
        30 => ("BF16", 2, 1),
        _ => ("UNKNOWN", 0, 1),
    }
}

pub fn inspect(b: &[u8]) -> Result<Value, String> {
    let mut r = R { b, i: 0 };
    if r.take(4)? != b"GGUF" {
        return Err("not a GGUF file (bad magic)".into());
    }
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(format!("unsupported GGUF version {version}"));
    }
    let n_tensors = r.u64()?;
    let n_kv = r.u64()?;
    if n_tensors > 100_000 || n_kv > 100_000 {
        return Err("implausible tensor / metadata counts".into());
    }
    let mut meta = Map::new();
    for _ in 0..n_kv {
        let k = r.string()?;
        let t = r.u32()?;
        meta.insert(k, r.value(t, 0)?);
    }
    let mut by_type: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut params: u64 = 0;
    let mut bytes: u64 = 0;
    let mut tensors = Vec::new();
    for _ in 0..n_tensors {
        let name = r.string()?;
        let nd = r.u32()?;
        if nd > 4 {
            return Err(format!("tensor {name} has {nd} dims"));
        }
        let dims: Vec<u64> = (0..nd).map(|_| r.u64()).collect::<Result<_, _>>()?;
        let t = r.u32()?;
        let _offset = r.u64()?;
        let n: u64 = dims.iter().product();
        let (tn, bpb, epb) = ggml_type(t);
        let sz = n.div_ceil(epb) * bpb;
        params += n;
        bytes += sz;
        let e = by_type.entry(tn.to_string()).or_default();
        e.0 += 1;
        e.1 += n;
        if tensors.len() < 12 {
            tensors.push(json!({"name": name, "dims": dims, "type": tn, "bytes": sz}));
        }
    }
    let arch = meta
        .get("general.architecture")
        .and_then(Value::as_str)
        .unwrap_or("?")
        .to_string();
    let get = |k: &str| {
        meta.get(&format!("{arch}.{k}"))
            .cloned()
            .unwrap_or(Value::Null)
    };
    let mix: Map<String, Value> = by_type
        .iter()
        .map(|(k, (c, n))| (k.clone(), json!({"tensors": c, "share_of_params": (*n as f64 / params.max(1) as f64 * 1000.0).round() / 1000.0})))
        .collect();
    Ok(json!({
        "version": version, "architecture": arch,
        "name": meta.get("general.name"),
        "context_length": get("context_length"), "block_count": get("block_count"),
        "embedding_length": get("embedding_length"), "head_count": get("attention.head_count"),
        "head_count_kv": get("attention.head_count_kv"),
        "parameters": params, "parameters_b": (params as f64 / 1e8).round() / 10.0,
        "tensor_bytes": bytes, "bits_per_weight": ((bytes * 8) as f64 / params.max(1) as f64 * 100.0).round() / 100.0,
        "quant_mix": mix, "tensors_preview": tensors, "metadata": meta, "header_bytes": r.i
    }))
}

pub fn inspect_op(i: &Value) -> Result<Value, String> {
    let h = i["hex"]
        .as_str()
        .ok_or("missing `hex` (the first bytes of the file, hex-encoded)")?;
    let h: String = h.chars().filter(|c| !c.is_whitespace()).collect();
    if h.len() % 2 == 1 {
        return Err("odd hex length".into());
    }
    let b: Vec<u8> = (0..h.len())
        .step_by(2)
        .map(|k| u8::from_str_radix(&h[k..k + 2], 16).map_err(|_| "bad hex"))
        .collect::<Result<_, _>>()?;
    inspect(&b)
}

/// Build a tiny GGUF header (used by tests and the demo).
pub fn build_sample(kv: &[(&str, Value)], tensors: &[(&str, Vec<u64>, u32)]) -> Vec<u8> {
    let mut b = b"GGUF".to_vec();
    b.extend(3u32.to_le_bytes());
    b.extend((tensors.len() as u64).to_le_bytes());
    b.extend((kv.len() as u64).to_le_bytes());
    let s = |b: &mut Vec<u8>, x: &str| {
        b.extend((x.len() as u64).to_le_bytes());
        b.extend(x.as_bytes());
    };
    for (k, v) in kv {
        s(&mut b, k);
        match v {
            Value::String(x) => {
                b.extend(8u32.to_le_bytes());
                s(&mut b, x);
            }
            Value::Number(n) => {
                b.extend(4u32.to_le_bytes());
                b.extend((n.as_u64().unwrap() as u32).to_le_bytes());
            }
            _ => panic!("sample supports strings and u32"),
        }
    }
    let mut off = 0u64;
    for (name, dims, t) in tensors {
        s(&mut b, name);
        b.extend((dims.len() as u32).to_le_bytes());
        for d in dims {
            b.extend(d.to_le_bytes());
        }
        b.extend(t.to_le_bytes());
        b.extend(off.to_le_bytes());
        off += 1024;
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Vec<u8> {
        build_sample(
            &[
                ("general.architecture", json!("llama")),
                ("general.name", json!("tiny")),
                ("llama.context_length", json!(4096)),
            ],
            &[
                ("token_embd.weight", vec![4096, 32000], 14),
                ("blk.0.attn_q.weight", vec![4096, 4096], 12),
                ("output_norm.weight", vec![4096], 0),
            ],
        )
    }
    #[test]
    fn reads_header() {
        let r = inspect(&sample()).unwrap();
        assert_eq!(r["architecture"], "llama");
        assert_eq!(r["context_length"], 4096);
        assert_eq!(r["parameters"], 4096 * 32000 + 4096 * 4096 + 4096);
        assert_eq!(r["quant_mix"]["Q4_K"]["tensors"], 1);
    }
    #[test]
    fn truncated_is_an_error_not_a_panic() {
        let s = sample();
        for cut in 0..s.len() {
            assert!(inspect(&s[..cut]).is_err());
        }
    }
    #[test]
    fn bad_magic() {
        assert!(inspect(b"GGML\x03\x00\x00\x00").is_err());
    }
}
