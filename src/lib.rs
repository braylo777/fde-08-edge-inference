//! Stage 8 — Edge & On-Prem Local Inference.
//!
//! * `gguf`  — read the header of a GGUF model file (llama.cpp format): metadata, tensor
//!   table, quantization mix, parameter count and size on disk.
//! * `quant` — int4 group-wise round-to-nearest quantization and AWQ-style activation-aware
//!   scaling, measured on a synthetic layer with outlier input channels.
//! * `plan`  — will this model fit on this GPU / laptop? Weights + KV cache + overhead for each
//!   quantization, and the best one that fits.
//! * `route` — offline fallback routing with local triage and a PII screen that runs before
//!   any network decision.

mod api;
pub mod gguf;
pub mod plan;
pub mod quant;
pub mod route;

pub use api::dispatch;
use serde_json::Value;

pub const OPS: &[&str] = &["gguf_inspect", "quantize_compare", "memory_plan", "route"];

pub fn handle(op: &str, i: &Value) -> Result<Value, String> {
    match op {
        "gguf_inspect" => gguf::inspect_op(i),
        "quantize_compare" => quant::compare_op(i),
        "memory_plan" => plan::plan_op(i),
        "route" => route::route_op(i),
        _ => Err(format!("unknown op `{op}`; known: {}", OPS.join(", "))),
    }
}
