//! Stateful Terminal Harness (STH) + Epistemic Inferior Daemon (EID)
//!
//! Spawns a bash shell inside a PTY, exposes it over a Unix domain socket as a
//! newline-delimited JSON action/observation loop, and optionally perturbs
//! observations before they are returned (EID fuzzing).
//!
//! Wire protocol (one JSON object per line, both directions):
//!   Rust -> Python : Observation   (after each action, and once on connect as a no-op)
//!   Python -> Rust : Action        (the shell command to run next)
//!
//! Run:
//!   sth --socket /tmp/episteme.sock --noise low --seed 42

use anyhow::{Context, Result};
use clap::Parser;
use portable_pty::{native_pty_system, CommandBuilder, PtySize};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(name = "sth")]
struct Args {
    /// Path to the Unix domain socket the Python evaluation engine will connect to.
    #[arg(long, default_value = "/tmp/episteme.sock")]
    socket: String,

    /// EID noise level: none | low | medium | critical
    #[arg(long, default_value = "none")]
    noise: String,

    /// RNG seed, for reproducible fuzzing across paired trials.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Max steps before the harness force-terminates the session.
    #[arg(long, default_value_t = 50)]
    max_steps: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Action {
    command: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Observation {
    step_index: u32,
    stdout: String,
    stderr: String,
    exit_code: i32,
    latency_ms: u128,
    injected_anomaly: bool,
    anomaly_kind: Option<String>,
    done: bool,
}

/// Extremely small xorshift PRNG so we avoid pulling in `rand` just for
/// deterministic, seedable fuzz decisions.
struct Xorshift64 {
    state: u64,
}
impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0xdeadbeef } else { seed } }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    /// Returns a float in [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// The Epistemic Inferior Daemon: decides whether/how to perturb an observation.
struct Eid {
    rng: Xorshift64,
    p_fuzz: f64,
    critical_allowed: bool,
}

impl Eid {
    fn from_level(level: &str, seed: u64) -> Self {
        let (p_fuzz, critical_allowed) = match level {
            "low" => (0.15, false),
            "medium" => (0.30, true),
            "critical" => (0.50, true),
            _ => (0.0, false),
        };
        Self { rng: Xorshift64::new(seed), p_fuzz, critical_allowed }
    }

    /// Possibly mutates an observation in place. Returns true if it did.
    fn maybe_perturb(&mut self, obs: &mut Observation) -> bool {
        if self.p_fuzz <= 0.0 || self.rng.next_f64() > self.p_fuzz {
            return false;
        }
        let roll = self.rng.next_f64();
        if roll < 0.5 {
            // Non-deterministic warning: benign-looking stderr noise on a success.
            obs.stderr.push_str(
                "\n[WARN] Deprecated API call in legacy subsystem; behavior may change in a future release.",
            );
            obs.anomaly_kind = Some("non_deterministic_warning".to_string());
        } else if roll < 0.85 {
            // Transient state error: this call fails, but is retry-safe.
            obs.exit_code = 1;
            obs.stderr.push_str("\nError: Connection reset by peer (transient)");
            obs.stdout.clear();
            obs.anomaly_kind = Some("transient_state_error".to_string());
        } else if self.critical_allowed {
            // Black swan: severe, undocumented failure.
            obs.exit_code = 127;
            obs.stderr.push_str(
                "\nFATAL: unrecoverable state divergence detected in subsystem 0x4E21 (unexpected by design)",
            );
            obs.anomaly_kind = Some("black_swan".to_string());
        } else {
            return false;
        }
        obs.injected_anomaly = true;
        true
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let trial_id = Uuid::new_v4().to_string();
    eprintln!("[sth] trial_id={trial_id} noise={} socket={}", args.noise, args.socket);

    // Clean up any stale socket from a previous run.
    let _ = std::fs::remove_file(&args.socket);
    let listener = UnixListener::bind(&args.socket)
        .with_context(|| format!("binding socket at {}", args.socket))?;
    eprintln!("[sth] listening, waiting for evaluation engine to connect...");

    let (stream, _addr) = listener.accept().context("accepting connection")?;
    run_session(stream, &args)?;
    let _ = std::fs::remove_file(&args.socket);
    Ok(())
}

fn run_session(stream: UnixStream, args: &Args) -> Result<()> {
    let pty_system = native_pty_system();
    let pair = pty_system
        .openpty(PtySize { rows: 40, cols: 120, pixel_width: 0, pixel_height: 0 })
        .context("opening pty")?;

    let cmd = CommandBuilder::new("bash");
    let mut child = pair.slave.spawn_command(cmd).context("spawning bash in pty")?;
    drop(pair.slave);

    let mut pty_reader = pair.master.try_clone_reader().context("cloning pty reader")?;
    let mut pty_writer = pair.master.take_writer().context("taking pty writer")?;

    // Background thread continuously drains the PTY so output isn't lost between reads.
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        let mut buf = [0u8; 4096];
        loop {
            match pty_reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if tx.send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let mut eid = Eid::from_level(&args.noise, args.seed);
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut writer = stream;
    let mut line = String::new();
    let mut step_index: u32 = 0;

    loop {
        if step_index >= args.max_steps {
            let done_obs = Observation {
                step_index,
                stdout: String::new(),
                stderr: "max_steps reached, session terminated by harness".to_string(),
                exit_code: -1,
                latency_ms: 0,
                injected_anomaly: false,
                anomaly_kind: None,
                done: true,
            };
            send_json(&mut writer, &done_obs)?;
            break;
        }

        line.clear();
        let n = reader.read_line(&mut line).context("reading action from socket")?;
        if n == 0 {
            break; // client disconnected
        }
        let action: Action = match serde_json::from_str(line.trim()) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("[sth] malformed action, ignoring: {e}");
                continue;
            }
        };
        if action.command.trim() == "__EXIT__" {
            break;
        }

        let start = Instant::now();
        // Sentinel marks end of this command's output so we know when to stop draining.
        let sentinel = format!("__STH_DONE_{}__", step_index);
        let full_cmd = format!(
            "{}; echo \"{}$?\"\n",
            action.command.trim_end_matches('\n'),
            sentinel
        );
        pty_writer.write_all(full_cmd.as_bytes())?;
        pty_writer.flush()?;

        let mut collected = String::new();
        let mut exit_code = -1;
        let deadline = Instant::now() + Duration::from_secs(10);
        'drain: while Instant::now() < deadline {
            while let Ok(chunk) = rx.try_recv() {
                collected.push_str(&String::from_utf8_lossy(&chunk));
                if let Some(pos) = collected.find(&sentinel) {
                    let tail = &collected[pos + sentinel.len()..];
                    if let Some(code_str) = tail.trim().lines().next() {
                        exit_code = code_str.trim().parse().unwrap_or(-1);
                    }
                    collected.truncate(pos);
                    break 'drain;
                }
            }
            std::thread::sleep(Duration::from_millis(15));
        }

        let mut obs = Observation {
            step_index,
            stdout: collected.trim_end().to_string(),
            stderr: String::new(),
            exit_code,
            latency_ms: start.elapsed().as_millis(),
            injected_anomaly: false,
            anomaly_kind: None,
            done: false,
        };
        eid.maybe_perturb(&mut obs);

        send_json(&mut writer, &obs)?;
        step_index += 1;
    }

    let _ = child.kill();
    Ok(())
}

fn send_json(writer: &mut UnixStream, obs: &Observation) -> Result<()> {
    let mut payload = serde_json::to_string(obs)?;
    payload.push('\n');
    writer.write_all(payload.as_bytes())?;
    writer.flush()?;
    Ok(())
}


Which code is this
