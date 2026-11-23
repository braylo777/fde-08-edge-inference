//! Memory fit planner: weights + KV cache + runtime overhead, per quantization.

use serde_json::{json, Value};

/// (name, effective bits per weight incl. scales) — llama.cpp / vLLM common formats.
pub const QUANTS: &[(&str, f64)] = &[
    ("F16", 16.0),
    ("Q8_0", 8.5),
    ("Q6_K", 6.56),
    ("Q5_K_M", 5.69),
    ("Q4_K_M", 4.85),
    ("AWQ-int4 (g128)", 4.25),
    ("Q3_K_M", 3.91),
    ("Q2_K", 3.35),
];

pub fn plan_op(i: &Value) -> Result<Value, String> {
    let m = &i["model"];
    let params_b = m["params_b"]
        .as_f64()
        .ok_or("model.params_b (billions) is required")?;
    let layers = m["layers"].as_f64().ok_or("model.layers is required")?;
    let kv_heads = m["kv_heads"].as_f64().ok_or("model.kv_heads is required")?;
    let head_dim = m["head_dim"].as_f64().ok_or("model.head_dim is required")?;
    let ctx = i["context_tokens"].as_f64().unwrap_or(8192.0);
    let batch = i["concurrent_sequences"].as_f64().unwrap_or(1.0);
    let kv_bytes = match i["kv_cache_dtype"].as_str().unwrap_or("f16") {
        "f16" | "bf16" => 2.0,
        "q8_0" | "fp8" => 1.0,
        "q4_0" => 0.5625,
        other => {
            return Err(format!(
                "kv_cache_dtype {other} not known (f16, fp8, q8_0, q4_0)"
            ))
        }
    };
    let device_gb = i["device_memory_gb"]
        .as_f64()
        .ok_or("device_memory_gb is required")?;
    let reserve = i["reserve_fraction"].as_f64().unwrap_or(0.10);
    let overhead_gb = i["runtime_overhead_gb"].as_f64().unwrap_or(0.8);
    let gb = 1e9;
    let kv_gb = 2.0 * layers * kv_heads * head_dim * ctx * batch * kv_bytes / gb;
    let usable = device_gb * (1.0 - reserve);
    let mut rows = Vec::new();
    let mut best: Option<&str> = None;
    for (name, bits) in QUANTS {
        let w = params_b * 1e9 * bits / 8.0 / gb;
        let total = w + kv_gb + overhead_gb;
        let fits = total <= usable;
        if fits && best.is_none() {
            best = Some(name);
        }
        rows.push(json!({"quant": name, "weights_gb": (w * 100.0).round() / 100.0, "total_gb": (total * 100.0).round() / 100.0,
            "fits": fits, "headroom_gb": ((usable - total) * 100.0).round() / 100.0}));
    }
    let max_ctx_best = best.map(|b| {
        let bits = QUANTS.iter().find(|q| q.0 == b).unwrap().1;
        let free = usable - params_b * 1e9 * bits / 8.0 / gb - overhead_gb;
        (free * gb / (2.0 * layers * kv_heads * head_dim * batch * kv_bytes)).floor()
    });
    Ok(json!({
        "kv_cache_gb": (kv_gb * 100.0).round() / 100.0, "usable_gb": (usable * 100.0).round() / 100.0,
        "options": rows,
        "recommendation": match best {
            Some(b) => format!("{b}: highest quality that fits; max context at this batch ≈ {} tokens", max_ctx_best.unwrap()),
            None => "nothing fits: reduce context or concurrency, quantize the KV cache, or add a GPU".into()
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn llama8b(dev: f64, ctx: f64) -> Value {
        plan_op(
            &json!({"model": {"params_b": 8.0, "layers": 32, "kv_heads": 8, "head_dim": 128},
            "context_tokens": ctx, "device_memory_gb": dev}),
        )
        .unwrap()
    }
    #[test]
    fn kv_cache_math() {
        // 2 * 32 layers * 8 heads * 128 dim * 8192 tokens * 2 bytes = 1.07 GB
        assert_eq!(llama8b(24.0, 8192.0)["kv_cache_gb"], 1.07);
    }
    #[test]
    fn fits_f16_on_24gb() {
        assert!(llama8b(24.0, 8192.0)["recommendation"]
            .as_str()
            .unwrap()
            .starts_with("F16"));
    }
    #[test]
    fn needs_4bit_on_8gb() {
        let r = llama8b(8.0, 4096.0);
        assert!(
            r["recommendation"].as_str().unwrap().starts_with("Q5_K_M")
                || r["recommendation"].as_str().unwrap().starts_with("Q4_K_M")
        );
    }
    #[test]
    fn nothing_fits_on_2gb() {
        assert!(llama8b(2.0, 8192.0)["recommendation"]
            .as_str()
            .unwrap()
            .starts_with("nothing"));
    }
}
