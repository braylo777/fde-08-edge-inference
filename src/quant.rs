//! Int4 quantization: plain round-to-nearest (RTN) vs. AWQ-style activation-aware scaling.
//!
//! AWQ idea (Lin et al., 2023): a few input channels carry large activations, so their weights
//! matter most. Scale those weight columns up by s_j = mean|x_j|^alpha before quantizing and
//! divide back after: the salient weights get finer quantization steps. alpha is grid-searched
//! to minimize the *output* error on calibration activations.

use serde_json::{json, Value};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn gauss(&mut self) -> f64 {
        let (u, v) = (self.next().max(1e-12), self.next());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Symmetric int4 per group along the input dim: q in [-8, 7], scale = max|w| / 7.
pub fn quant_rtn(w: &[f64], cols: usize, group: usize) -> Vec<f64> {
    let mut out = vec![0.0; w.len()];
    for (r, row) in w.chunks(cols).enumerate() {
        for (g, chunk) in row.chunks(group).enumerate() {
            let m = chunk.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            let scale = if m == 0.0 { 1.0 } else { m / 7.0 };
            for (k, x) in chunk.iter().enumerate() {
                let q = (x / scale).round().clamp(-8.0, 7.0);
                out[r * cols + g * group + k] = q * scale;
            }
        }
    }
    out
}

/// y = X W^T ; X: n×cols, W: rows×cols
fn matmul(x: &[f64], n: usize, w: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut y = vec![0.0; n * rows];
    for i in 0..n {
        for r in 0..rows {
            y[i * rows + r] = (0..cols).map(|c| x[i * cols + c] * w[r * cols + c]).sum();
        }
    }
    y
}

fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    let num: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
    let den: f64 = a.iter().map(|x| x * x).sum();
    (num / den).sqrt()
}

pub fn awq(
    w: &[f64],
    x: &[f64],
    n: usize,
    rows: usize,
    cols: usize,
    group: usize,
) -> (Vec<f64>, f64, Vec<(f64, f64)>) {
    let y = matmul(x, n, w, rows, cols);
    let act: Vec<f64> = (0..cols)
        .map(|c| (0..n).map(|i| x[i * cols + c].abs()).sum::<f64>() / n as f64)
        .collect();
    let mut best = (f64::MAX, 0.0, w.to_vec());
    let mut curve = Vec::new();
    for step in 0..=8 {
        let alpha = step as f64 / 8.0;
        let s: Vec<f64> = act.iter().map(|a| a.max(1e-6).powf(alpha)).collect();
        // normalize so the geometric mean scale is 1 (keeps magnitudes sane)
        let gm = (s.iter().map(|v| v.ln()).sum::<f64>() / cols as f64).exp();
        let s: Vec<f64> = s.iter().map(|v| v / gm).collect();
        let scaled: Vec<f64> = w.iter().enumerate().map(|(k, v)| v * s[k % cols]).collect();
        let deq: Vec<f64> = quant_rtn(&scaled, cols, group)
            .iter()
            .enumerate()
            .map(|(k, v)| v / s[k % cols])
            .collect();
        let e = rel_err(&y, &matmul(x, n, &deq, rows, cols));
        curve.push((alpha, (e * 1e5).round() / 1e5));
        if e < best.0 {
            best = (e, alpha, deq);
        }
    }
    (best.2, best.1, curve)
}

pub fn compare_op(i: &Value) -> Result<Value, String> {
    let rows = i["rows"].as_u64().unwrap_or(64) as usize;
    let cols = i["cols"].as_u64().unwrap_or(256) as usize;
    let n = i["samples"].as_u64().unwrap_or(64) as usize;
    let group = i["group_size"].as_u64().unwrap_or(32) as usize;
    let outliers = i["outlier_channels"].as_u64().unwrap_or(4) as usize;
    let mag = i["outlier_magnitude"].as_f64().unwrap_or(20.0);
    if rows * cols > 200_000
        || n * cols > 200_000
        || group == 0
        || cols % group != 0
        || outliers > cols
    {
        return Err(
            "keep rows*cols and samples*cols under 200k; cols must be a multiple of group_size"
                .into(),
        );
    }
    let mut rng = Rng(i["seed"].as_u64().unwrap_or(7).max(1));
    let w: Vec<f64> = (0..rows * cols).map(|_| rng.gauss() * 0.02).collect();
    let hot: Vec<usize> = (0..outliers).map(|k| (k * 37 + 5) % cols).collect();
    let x: Vec<f64> = (0..n * cols)
        .map(|k| rng.gauss() * if hot.contains(&(k % cols)) { mag } else { 1.0 })
        .collect();
    let y = matmul(&x, n, &w, rows, cols);
    let rtn = quant_rtn(&w, cols, group);
    let e_rtn = rel_err(&y, &matmul(&x, n, &rtn, rows, cols));
    let (_, alpha, curve) = awq(&w, &x, n, rows, cols, group);
    let e_awq = curve.iter().map(|c| c.1).fold(f64::MAX, f64::min);
    let bits = 4.0 + 16.0 / group as f64;
    Ok(json!({
        "layer": {"rows": rows, "cols": cols, "group_size": group, "salient_channels": hot},
        "bits_per_weight": bits, "compression_vs_fp16": (16.0 / bits * 100.0).round() / 100.0,
        "output_rel_error": {"rtn_int4": (e_rtn * 1e5).round() / 1e5, "awq_int4": e_awq},
        "error_reduction": format!("{:.1}%", (1.0 - e_awq / e_rtn) * 100.0),
        "best_alpha": alpha, "alpha_sweep": curve.iter().map(|(a, e)| json!({"alpha": a, "error": e})).collect::<Vec<_>>(),
        "note": "alpha 0 is plain RTN; error is ||XW^T - XW_q^T|| / ||XW^T|| on the calibration batch"
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rtn_levels_and_bound() {
        let w = vec![0.7, -0.35, 0.1, 0.0, 0.69, -0.7, 0.2, 0.05];
        let q = quant_rtn(&w, 8, 4);
        for (a, b) in w.iter().zip(&q) {
            assert!((a - b).abs() <= 0.7 / 7.0 / 2.0 + 1e-12);
        }
    }
    #[test]
    fn zero_group_is_safe() {
        assert_eq!(quant_rtn(&[0.0; 4], 4, 4), vec![0.0; 4]);
    }
    #[test]
    fn awq_beats_rtn_with_outliers() {
        let r = compare_op(&json!({"seed": 3})).unwrap();
        let rtn = r["output_rel_error"]["rtn_int4"].as_f64().unwrap();
        let awq = r["output_rel_error"]["awq_int4"].as_f64().unwrap();
        assert!(awq < rtn * 0.9, "awq {awq} rtn {rtn}");
        assert!(r["best_alpha"].as_f64().unwrap() > 0.0);
    }
    #[test]
    fn awq_never_worse_than_rtn() {
        let r = compare_op(&json!({"seed": 5, "outlier_channels": 0})).unwrap();
        assert!(
            r["output_rel_error"]["awq_int4"].as_f64().unwrap()
                <= r["output_rel_error"]["rtn_int4"].as_f64().unwrap() + 1e-9
        );
    }
}
