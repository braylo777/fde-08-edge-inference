//! Offline-first routing for on-prem / edge deployments.
//!
//! Order of checks (cheapest and safest first):
//! 1. PII screen (microseconds, local): any hit pins the request to local targets only.
//! 2. Local triage: a rule-based classifier scores complexity from the text.
//! 3. Target choice: the smallest healthy target that meets complexity, latency and data rules.
//!    If the network is down or a target is unhealthy, fall back down the list; if nothing can
//!    serve the request, queue it rather than send data where policy forbids.

use serde_json::{json, Value};

pub fn pii_hits(t: &str) -> Vec<&'static str> {
    let mut hits = Vec::new();
    if t.split_whitespace()
        .any(|w| w.contains('@') && w.contains('.'))
    {
        hits.push("email");
    }
    let digits_runs: Vec<usize> = t
        .split(|c: char| !(c.is_ascii_digit() || c == '-' || c == ' '))
        .map(|s| s.chars().filter(|c| c.is_ascii_digit()).count())
        .collect();
    if digits_runs.iter().any(|&n| (13..=19).contains(&n)) {
        hits.push("card-like number");
    }
    if digits_runs.contains(&9) && t.contains('-') {
        hits.push("ssn-like number");
    }
    let low = t.to_lowercase();
    for (w, tag) in [
        ("diagnos", "health"),
        ("patient", "health"),
        ("salary", "hr"),
        ("passport", "id document"),
        ("ssn", "ssn"),
    ] {
        if low.contains(w) && !hits.contains(&tag) {
            hits.push(tag);
        }
    }
    hits
}

/// 1 = trivial (classify / extract), 2 = moderate (summarize / draft), 3 = hard (multi-step reasoning, code, long docs).
pub fn triage(t: &str) -> (u8, Vec<String>) {
    let low = t.to_lowercase();
    let words = t.split_whitespace().count();
    let mut why = vec![];
    let mut level = 1u8;
    for (k, lvl) in [
        ("classify", 1),
        ("extract", 1),
        ("yes or no", 1),
        ("summarize", 2),
        ("draft", 2),
        ("rewrite", 2),
        ("translate", 2),
        ("analyze", 3),
        ("compare", 3),
        ("plan", 3),
        ("step by step", 3),
        ("write code", 3),
        ("prove", 3),
        ("why", 3),
    ] {
        if low.contains(k) && lvl > level {
            level = lvl;
            why.push(format!("keyword `{k}`"));
        }
    }
    if words > 400 && level < 3 {
        level += 1;
        why.push(format!("{words} words"));
    }
    if why.is_empty() {
        why.push("short request, no complex keywords".into());
    }
    (level, why)
}

pub fn route_op(i: &Value) -> Result<Value, String> {
    let text = i["text"].as_str().ok_or("missing `text`")?;
    let online = i["network_up"].as_bool().unwrap_or(true);
    let budget = i["latency_budget_ms"].as_u64().unwrap_or(10_000);
    let targets = i["targets"].as_array().ok_or("`targets` is required: [{name, location: device|onprem|cloud, max_complexity, p50_ms, healthy}]")?;
    let pii = pii_hits(text);
    let (level, why) = triage(text);
    let mut trace = Vec::new();
    for t in targets {
        let name = t["name"].as_str().unwrap_or("?");
        let loc = t["location"].as_str().unwrap_or("cloud");
        let reason = if t["healthy"] == false {
            Some("unhealthy".to_string())
        } else if loc == "cloud" && !pii.is_empty() {
            Some(format!("PII ({}) may not leave the site", pii.join(", ")))
        } else if loc != "device" && !online && loc == "cloud" {
            Some("network down".to_string())
        } else if t["max_complexity"].as_u64().unwrap_or(3) < level as u64 {
            Some(format!(
                "complexity {level} above its max {}",
                t["max_complexity"]
            ))
        } else if t["p50_ms"].as_u64().unwrap_or(0) > budget {
            Some(format!("p50 {} ms over budget {budget} ms", t["p50_ms"]))
        } else {
            None
        };
        match reason {
            None => {
                trace.push(json!({"target": name, "decision": "selected"}));
                return Ok(
                    json!({"route": name, "location": loc, "complexity": level, "triage": why, "pii": pii, "trace": trace}),
                );
            }
            Some(r) => trace.push(json!({"target": name, "decision": "skipped", "reason": r})),
        }
    }
    Ok(
        json!({"route": "queue", "location": "local queue", "complexity": level, "triage": why, "pii": pii, "trace": trace,
        "note": "no permitted target can serve this now; queued for retry instead of breaking data policy"}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn targets() -> Value {
        json!([
            {"name": "phi-mini-q4 (laptop)", "location": "device", "max_complexity": 1, "p50_ms": 300, "healthy": true},
            {"name": "llama-70b-awq (on-prem)", "location": "onprem", "max_complexity": 3, "p50_ms": 2500, "healthy": true},
            {"name": "frontier-api", "location": "cloud", "max_complexity": 3, "p50_ms": 1800, "healthy": true}])
    }
    fn route(text: &str, extra: Value) -> Value {
        let mut i = json!({"text": text, "targets": targets()});
        for (k, v) in extra.as_object().unwrap() {
            i[k] = v.clone();
        }
        route_op(&i).unwrap()
    }
    #[test]
    fn simple_goes_to_device() {
        assert_eq!(
            route("Classify this ticket: printer jammed", json!({}))["location"],
            "device"
        );
    }
    #[test]
    fn hard_goes_onprem_first() {
        assert_eq!(
            route("Analyze these contracts and compare risks", json!({}))["location"],
            "onprem"
        );
    }
    #[test]
    fn pii_never_goes_to_cloud() {
        let r = route(
            "Analyze the patient diagnosis for jo@x.com",
            json!({"targets": [{"name": "c", "location": "cloud", "healthy": true}]}),
        );
        assert_eq!(r["route"], "queue");
    }
    #[test]
    fn onprem_down_falls_back_to_cloud_without_pii() {
        let mut t = targets();
        t[1]["healthy"] = json!(false);
        assert_eq!(
            route(
                "Analyze market trends and compare vendors",
                json!({"targets": t})
            )["location"],
            "cloud"
        );
    }
    #[test]
    fn offline_and_onprem_down_queues() {
        let mut t = targets();
        t[1]["healthy"] = json!(false);
        assert_eq!(
            route(
                "Plan the migration step by step",
                json!({"targets": t, "network_up": false})
            )["route"],
            "queue"
        );
    }
}
