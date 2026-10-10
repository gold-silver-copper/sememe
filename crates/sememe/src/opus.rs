//! The single wrapper for every model call: `claude -p --model
//! claude-opus-5-5`, with a shared concurrency limit, timeouts, backoff,
//! usage-limit pauses and a ledger row per call.

use std::io::{Read as _, Write as _};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The only model used, for every role.
pub const MODEL: &str = "claude-opus-5-5";

/// Which tier made a call.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// The teacher (medium tier).
    Teacher,
    /// Agents (slow tier).
    Agent,
    /// The start-up self-test.
    SelfTest,
}

/// One ledger row.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LedgerRow {
    /// Unix seconds.
    pub ts: u64,
    /// Tier.
    pub tier: String,
    /// Round.
    pub round: u32,
    /// Input tokens (uncached).
    pub input: u64,
    /// Cache-creation input tokens.
    pub cache_write: u64,
    /// Cache-read input tokens.
    pub cache_read: u64,
    /// Output tokens.
    pub output: u64,
    /// `ok`, `error`, `timeout`, `limit`.
    pub outcome: String,
    /// Wall-clock seconds.
    pub secs: f64,
    /// What the call was for.
    pub purpose: String,
}

struct Slots {
    teacher: usize,
    agent: usize,
    paused_until: Option<Instant>,
}

/// The shared call wrapper.
pub struct Opus {
    ledger: PathBuf,
    workdir: PathBuf,
    slots: Mutex<Slots>,
    cv: Condvar,
    max_teacher: usize,
    max_agent: usize,
}

/// The result of a structured call.
pub struct Reply {
    /// The structured output (or parsed text).
    pub value: Value,
    /// The ledger row written for it.
    pub row: LedgerRow,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl Opus {
    /// A wrapper writing its ledger to `ledger`; calls run in `workdir`
    /// (outside the repo, so no project instructions are loaded).
    pub fn new(ledger: &Path, workdir: &Path) -> Result<Opus> {
        std::fs::create_dir_all(workdir)?;
        if let Some(d) = ledger.parent() {
            std::fs::create_dir_all(d)?;
        }
        Ok(Opus {
            ledger: ledger.to_path_buf(),
            workdir: workdir.to_path_buf(),
            slots: Mutex::new(Slots {
                teacher: 0,
                agent: 0,
                paused_until: None,
            }),
            cv: Condvar::new(),
            max_teacher: 8,
            max_agent: 4,
        })
    }

    fn acquire(&self, tier: Tier) {
        let Ok(mut s) = self.slots.lock() else { return };
        loop {
            if let Some(until) = s.paused_until {
                let now = Instant::now();
                if now < until {
                    let wait = until - now;
                    match self.cv.wait_timeout(s, wait) {
                        Ok((g, _)) => s = g,
                        Err(_) => return,
                    }
                    continue;
                }
                s.paused_until = None;
            }
            let free = match tier {
                Tier::Agent => s.agent < self.max_agent,
                _ => s.teacher < self.max_teacher,
            };
            if free {
                match tier {
                    Tier::Agent => s.agent += 1,
                    _ => s.teacher += 1,
                }
                return;
            }
            match self.cv.wait(s) {
                Ok(g) => s = g,
                Err(_) => return,
            }
        }
    }

    fn release(&self, tier: Tier) {
        if let Ok(mut s) = self.slots.lock() {
            match tier {
                Tier::Agent => s.agent = s.agent.saturating_sub(1),
                _ => s.teacher = s.teacher.saturating_sub(1),
            }
        }
        self.cv.notify_all();
    }

    fn pause(&self, secs: u64) {
        if let Ok(mut s) = self.slots.lock() {
            s.paused_until = Some(Instant::now() + Duration::from_secs(secs));
        }
        eprintln!("opus: usage limit; pausing all tiers for {secs}s");
    }

    fn write_row(&self, row: &LedgerRow) {
        if let Ok(line) = serde_json::to_string(row)
            && let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&self.ledger)
        {
            let _ = writeln!(f, "{line}");
        }
    }

    /// A single-turn structured call with no tools: the teacher's kind of
    /// call. Retries with backoff on errors; waits out usage limits.
    #[allow(clippy::too_many_arguments)]
    pub fn structured(
        &self,
        tier: Tier,
        round: u32,
        purpose: &str,
        system: &str,
        prompt: &str,
        schema: &Value,
        timeout: Duration,
    ) -> Result<Reply> {
        let mut backoff = 30u64;
        for attempt in 0..6 {
            self.acquire(tier);
            let started = Instant::now();
            let out = run_claude(&self.workdir, system, prompt, schema, timeout);
            self.release(tier);
            let secs = started.elapsed().as_secs_f64();
            let mut row = LedgerRow {
                ts: now_secs(),
                tier: format!("{tier:?}").to_lowercase(),
                round,
                secs,
                purpose: purpose.to_string(),
                ..LedgerRow::default()
            };
            match out {
                Ok(v) => {
                    let usage = &v.get("usage").unwrap_or(&Value::Null);
                    row.input = usage
                        .get("input_tokens")
                        .unwrap_or(&Value::Null)
                        .as_u64()
                        .unwrap_or(0);
                    row.cache_write = usage
                        .get("cache_creation_input_tokens")
                        .unwrap_or(&Value::Null)
                        .as_u64()
                        .unwrap_or(0);
                    row.cache_read = usage
                        .get("cache_read_input_tokens")
                        .unwrap_or(&Value::Null)
                        .as_u64()
                        .unwrap_or(0);
                    row.output = usage
                        .get("output_tokens")
                        .unwrap_or(&Value::Null)
                        .as_u64()
                        .unwrap_or(0);
                    let is_error = v
                        .get("is_error")
                        .unwrap_or(&Value::Null)
                        .as_bool()
                        .unwrap_or(false);
                    let text = v
                        .get("result")
                        .unwrap_or(&Value::Null)
                        .as_str()
                        .unwrap_or_default()
                        .to_string();
                    if is_error {
                        let limited = text.to_lowercase().contains("limit")
                            || v.get("api_error_status").unwrap_or(&Value::Null).as_u64()
                                == Some(429);
                        row.outcome = if limited {
                            "limit".into()
                        } else {
                            "error".into()
                        };
                        self.write_row(&row);
                        if limited {
                            self.pause(reset_secs(&text).unwrap_or(3600));
                        } else {
                            std::thread::sleep(Duration::from_secs(backoff));
                            backoff = (backoff * 2).min(900);
                        }
                        eprintln!(
                            "opus: {purpose} attempt {attempt} failed: {}",
                            text.chars().take(300).collect::<String>()
                        );
                        continue;
                    }
                    let value = match v.get("structured_output") {
                        Some(s) if !s.is_null() => s.clone(),
                        _ => serde_json::from_str(&text)
                            .map_err(|e| anyhow!("unparsable reply: {e}"))?,
                    };
                    row.outcome = "ok".into();
                    self.write_row(&row);
                    return Ok(Reply { value, row });
                }
                Err(e) => {
                    row.outcome = if e.to_string().contains("timeout") {
                        "timeout".into()
                    } else {
                        "error".into()
                    };
                    self.write_row(&row);
                    eprintln!("opus: {purpose} attempt {attempt}: {e}");
                    std::thread::sleep(Duration::from_secs(backoff));
                    backoff = (backoff * 2).min(900);
                }
            }
        }
        Err(anyhow!("opus: {purpose} failed after retries"))
    }
}

/// Seconds until a reset time mentioned in a usage-limit message, if any.
fn reset_secs(text: &str) -> Option<u64> {
    // Messages carry a unix timestamp after a `|`, or "resets in N minutes".
    if let Some(ts) = text
        .split(|c: char| !c.is_ascii_digit())
        .filter(|s| s.len() == 10)
        .find_map(|s| s.parse::<u64>().ok())
    {
        let now = now_secs();
        if ts > now {
            return Some(ts - now + 60);
        }
    }
    None
}

fn run_claude(
    workdir: &Path,
    system: &str,
    prompt: &str,
    schema: &Value,
    timeout: Duration,
) -> Result<Value> {
    let mut child = Command::new("claude")
        .current_dir(workdir)
        .args([
            "-p",
            "--model",
            MODEL,
            "--tools",
            "",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--system-prompt",
            system,
            "--json-schema",
            &schema.to_string(),
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(prompt.as_bytes())?;
    }
    // Drain the pipes on threads so a large reply cannot block the child.
    let drain = |pipe: Option<Box<dyn std::io::Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut p) = pipe {
                let _ = p.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out_t = drain(
        child
            .stdout
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    );
    let err_t = drain(
        child
            .stderr
            .take()
            .map(|p| Box::new(p) as Box<dyn std::io::Read + Send>),
    );
    let started = Instant::now();
    loop {
        if child.try_wait()?.is_some() {
            break;
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(anyhow!("timeout after {}s", timeout.as_secs()));
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let stdout_bytes = out_t.join().unwrap_or_default();
    let stderr_bytes = err_t.join().unwrap_or_default();
    let stdout = String::from_utf8_lossy(&stdout_bytes);
    let json_line = stdout
        .lines()
        .rev()
        .find(|l| l.trim_start().starts_with('{'))
        .unwrap_or_default();
    serde_json::from_str(json_line).map_err(|e| {
        anyhow!(
            "bad CLI output ({e}): {} {}",
            stdout.chars().take(300).collect::<String>(),
            String::from_utf8_lossy(&stderr_bytes)
                .chars()
                .take(300)
                .collect::<String>()
        )
    })
}

/// The start-up self-test: one tiny call.
pub fn self_test(opus: &Opus) -> Result<()> {
    let schema = serde_json::json!({"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]});
    let r = opus.structured(
        Tier::SelfTest,
        0,
        "self-test",
        "Reply with JSON only.",
        "Return ok=true.",
        &schema,
        Duration::from_secs(120),
    )?;
    if r.value.get("ok").unwrap_or(&Value::Null).as_bool() == Some(true) {
        Ok(())
    } else {
        Err(anyhow!("self-test returned {}", r.value))
    }
}
