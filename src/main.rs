use futures_util::{SinkExt, StreamExt};
use prost::Message;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{ChildStdin, ChildStdout, Command};
use tokio::sync::{mpsc, Mutex as AsyncMutex};
use tokio_tungstenite::tungstenite::protocol::Message as WsMessage;
use tracing::{info, warn};
use tracing_subscriber::FmtSubscriber;

use cockatiel_client::proto::container_for_engine::Payload as EnginePayload;
use cockatiel_client::proto::container_for_module::Payload as ModulePayload;
use cockatiel_client::proto::*;
use cockatiel_client::{CockatielClient, PromptKind};

type WsWriteHalf = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>>,
    WsMessage,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Config {
    #[serde(default)]
    banned_words: Vec<String>,
    #[serde(default = "default_mode")]
    censor_mode: String,
    #[serde(default)]
    replace_word: String,
    #[serde(default)]
    flag_for_review: bool,
    /// "none" | "censor" | "replace" — apply to the WHOLE message/sentence
    /// instead of just the offending token.
    #[serde(default = "default_sentence_mode")]
    sentence_mode: String,
    #[serde(default)]
    replace_sentence: String,
    /// Master switch for the optional LLM review. When off (default) the module
    /// is purely the word-list detector/censor — no Python, no models.
    #[serde(default)]
    llm_review: bool,
    /// The "risk certainty" (0–1): a review score >= this is treated as a hit.
    #[serde(default = "default_review_threshold")]
    llm_review_threshold: f32,
    /// "deberta" (default) | "llama-guard" | "auto" — both optional.
    #[serde(default = "default_review_engine")]
    llm_review_engine: String,
    /// Round-trip timeout for a single LLM review (request-write +
    /// response-read), in seconds.
    #[serde(default = "default_review_timeout_secs")]
    review_timeout_secs: u32,
    /// Interpreter used to launch the review worker.
    #[serde(default = "default_python_interpreter")]
    python_interpreter: String,
    /// Path to the review worker script (relative to the module dir).
    #[serde(default = "default_worker_script_path")]
    worker_script_path: String,
    /// Timeout (seconds) for the operator setup prompts shown when no
    /// saved config exists.
    #[serde(default = "default_prompt_timeout_secs")]
    prompt_timeout_secs: u32,
    /// Reconnect backoff floor (seconds) when the engine connection drops.
    #[serde(default = "default_reconnect_base_secs")]
    reconnect_base_secs: u32,
    /// Reconnect backoff cap (seconds) after exponential growth.
    #[serde(default = "default_reconnect_max_secs")]
    reconnect_max_secs: u32,
    /// Worker model overrides (defaults keep the built-in worker behavior).
    #[serde(default = "default_llama_model")]
    llama_model: String,
    #[serde(default = "default_deberta_model")]
    deberta_model: String,
    #[serde(default = "default_deberta_max_length")]
    deberta_max_length: u32,
    #[serde(default = "default_llama_max_length")]
    llama_max_length: u32,
    #[serde(default = "default_llama_max_new_tokens")]
    llama_max_new_tokens: u32,
    /// Enable the lightweight in-process AI risk model (seed lexicon + shape
    /// signals). No Python, no model files, no network — just a 0-1 score.
    #[serde(default)]
    risk_model: bool,
    /// Risk-certainty threshold (0-1): a score >= this is treated as a hit.
    #[serde(default = "default_risk_threshold")]
    risk_threshold: f32,
}

fn default_risk_threshold() -> f32 {
    0.5
}

fn default_review_timeout_secs() -> u32 {
    10
}

fn default_python_interpreter() -> String {
    "python3".to_string()
}

fn default_worker_script_path() -> String {
    "worker/review_worker.py".to_string()
}

fn default_prompt_timeout_secs() -> u32 {
    60
}

fn default_reconnect_base_secs() -> u32 {
    1
}

fn default_reconnect_max_secs() -> u32 {
    30
}

fn default_llama_model() -> String {
    "meta-llama/Llama-Guard-3-1B".to_string()
}

fn default_deberta_model() -> String {
    "microsoft/deberta-v3-small".to_string()
}

fn default_deberta_max_length() -> u32 {
    256
}

fn default_llama_max_length() -> u32 {
    2000
}

fn default_llama_max_new_tokens() -> u32 {
    16
}

fn default_mode() -> String {
    "soft".to_string()
}

fn default_sentence_mode() -> String {
    "none".to_string()
}

fn default_review_threshold() -> f32 {
    0.5
}

fn default_review_engine() -> String {
    "deberta".to_string()
}

/// Normalise ONE character for matching: lowercase it, fold leet digits/symbols
/// to their letter (`4`→`a`, `@`→`a`, `$`→`s`, …), and fold common Latin
/// diacritics to the base letter (`ä`→`a`). This is what makes matching
/// resilient to "weird capitalization", leet, and accents.
fn fold(c: char) -> char {
    let c = match c.to_lowercase().next().unwrap_or(c) {
        '0' => 'o',
        '1' => 'i',
        '3' => 'e',
        '4' => 'a',
        '5' => 's',
        '7' => 't',
        '@' => 'a',
        '$' => 's',
        other => other,
    };
    // Common Latin diacritics → base letter. Not full Unicode normalisation —
    // just the letters a streamer realistically sees.
    match c {
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => 'a',
        'ç' | 'ć' | 'č' => 'c',
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ę' | 'ě' => 'e',
        'ì' | 'í' | 'î' | 'ï' | 'ī' => 'i',
        'ñ' | 'ń' => 'n',
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' => 'o',
        'ù' | 'ú' | 'û' | 'ü' | 'ū' => 'u',
        'ý' | 'ÿ' => 'y',
        'ß' => 's',
        other => other,
    }
}

/// A character is KEPT in the condensed form when, once folded, it is a letter
/// or digit. Everything else (spaces, punctuation, symbols) is a separator and
/// is dropped.
fn kept(c: char) -> bool {
    fold(c).is_alphanumeric()
}

/// A condensed view of a string: only the folded alphanumeric characters, in
/// order, each paired with its BYTE RANGE in the original. The condensed text
/// is what we match against; the spans let us censor the original in place and
/// tell a whole token from a substring.
struct Condensed {
    text: String,
    spans: Vec<(usize, usize)>,
}

/// Condense a string: lowercase + fold + drop every non-alphanumeric. `b a d w
/// o r d`, `badword!`, `BÄDWORD`, and `b4dw0rd` all condense to `badword`.
fn condense(s: &str) -> Condensed {
    let mut text = String::new();
    let mut spans = Vec::new();
    for (i, ch) in s.char_indices() {
        if kept(ch) {
            text.push(fold(ch));
            spans.push((i, i + ch.len_utf8()));
        }
    }
    Condensed { text, spans }
}

/// The text a module should operate on: the accumulated in-process text from
/// earlier modules, falling back to the raw chat message only when that is
/// empty. Censoring the raw message would discard earlier modules' edits.
fn effective_text<'a>(chat_raw: &'a str, incoming: &'a str) -> &'a str {
    if incoming.trim().is_empty() {
        chat_raw
    } else {
        incoming
    }
}

/// True when the condensed match of length `m` at condensed index `a` counts as
/// a hit: it spans a dropped separator (the space/punctuation-split evasion), OR
/// it is a whole token (bounded by separators/ends). A run sitting wholly inside
/// a longer word ("ass" in "class") is NOT a hit.
fn is_hit(msg: &str, c: &Condensed, a: usize, m: usize) -> bool {
    // An internal gap = a separator was dropped between two matched characters.
    for j in a..a + m - 1 {
        if c.spans[j + 1].0 != c.spans[j].1 {
            return true;
        }
    }
    // Whole token: the characters just outside the match are separators or ends.
    let before_ok = c.spans[a].0 == 0
        || msg[..c.spans[a].0]
            .chars()
            .next_back()
            .map_or(true, |ch| !kept(ch));
    let last = a + m - 1;
    let after_ok = c.spans[last].1 == msg.len()
        || msg[c.spans[last].1..]
            .chars()
            .next()
            .map_or(true, |ch| !kept(ch));
    before_ok && after_ok
}

/// The first banned word whose condensed form appears in `msg` as a hit, or
/// `None`. Words are tried in list order.
fn hit_word(msg: &str, banned: &[String]) -> Option<String> {
    let c = condense(msg);
    let chars: Vec<char> = c.text.chars().collect();
    let n = chars.len();
    for word in banned {
        let wchars: Vec<char> = condense(word).text.chars().collect();
        let m = wchars.len();
        if m == 0 || m > n {
            continue;
        }
        for a in 0..=n - m {
            if chars[a..a + m] == wchars[..] && is_hit(msg, &c, a, m) {
                return Some(word.clone());
            }
        }
    }
    None
}

/// Why `msg` matched `word`: "exact" when a whitespace-delimited token is the
/// word case-insensitively, else "normalized" (leet/split/accent). Logging only.
fn match_variant(msg: &str, word: &str) -> &'static str {
    let w = word.to_lowercase();
    if msg.split_whitespace().any(|t| t.to_lowercase() == w) {
        "exact"
    } else {
        "normalized"
    }
}

/// Check if the message contains any banned word and return the first match
/// together with why it matched. Matching is against the CONDENSED form (see
/// [`condense`]), so it tolerates spacing, punctuation, leet and accents.
fn detect_banned(message: &str, banned: &[String]) -> Option<(String, &'static str)> {
    hit_word(message, banned).map(|w| {
        let variant = match_variant(message, &w);
        (w, variant)
    })
}

/// Censor a single token (already known to contain a banned word).
fn censor_token(token: &str, mode: &str, replace_word: &str) -> String {
    match mode {
        "hard" => "*".repeat(token.chars().count()),
        "mid" => {
            let mut s = String::new();
            for (i, ch) in token.chars().enumerate() {
                s.push(if i == 0 { ch } else { '*' });
            }
            s
        }
        "replace" => {
            if replace_word.is_empty() {
                "*".repeat(token.chars().count())
            } else {
                replace_word.to_string()
            }
        }
        // "soft" (default): keep first + last letter, censor the middle.
        _ => {
            let chars: Vec<char> = token.chars().collect();
            if chars.len() <= 2 {
                "*".repeat(chars.len())
            } else {
                let mut s = String::new();
                s.push(chars[0]);
                for _ in 1..chars.len() - 1 {
                    s.push('*');
                }
                s.push(chars[chars.len() - 1]);
                s
            }
        }
    }
}

/// Censor any token in the message that matches a banned word (any variant).
/// Non-matching tokens are preserved exactly (including whitespace).
fn censor_message(message: &str, banned: &[String], mode: &str, replace_word: &str) -> String {
    let c = condense(message);
    let chars: Vec<char> = c.text.chars().collect();
    let n = chars.len();
    let words: Vec<Vec<char>> = banned
        .iter()
        .map(|w| condense(w).text.chars().collect::<Vec<char>>())
        .filter(|w| !w.is_empty())
        .collect();
    if words.is_empty() || n == 0 {
        return message.to_string();
    }
    // Collect non-overlapping hit spans as BYTE ranges in the original.
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let mut matched = false;
        for wchars in &words {
            let m = wchars.len();
            if i + m <= n && chars[i..i + m] == wchars[..] && is_hit(message, &c, i, m) {
                ranges.push((c.spans[i].0, c.spans[i + m - 1].1));
                i += m;
                matched = true;
                break;
            }
        }
        if !matched {
            i += 1;
        }
    }
    if ranges.is_empty() {
        return message.to_string();
    }
    // Rebuild, censoring only the matched byte ranges; everything else (spaces,
    // punctuation, case, accents) is preserved exactly.
    let mut out = String::new();
    let mut last = 0;
    for (s, e) in ranges {
        out.push_str(&message[last..s]);
        out.push_str(&censor_token(&message[s..e], mode, replace_word));
        last = e;
    }
    out.push_str(&message[last..]);
    out
}

// ---------------------------------------------------------------------------
// Lightweight in-process AI risk model
// ---------------------------------------------------------------------------
//
// A tiny logistic regression over a built-in seed lexicon plus obfuscation /
// shape signals. It returns a "risk" 0..1 in microseconds with NO Python, NO
// model file and NO network — the operator enables it and sets a threshold.
// This is intentionally small and explainable rather than a transformer.

/// Built-in seed lexicon (folded). Deliberately modest and generic — common
/// profanity/insults, no slurs. It exists so the model catches shapes the
/// operator's own `banned_words` list may miss. Tune freely.
const RISK_SEED: &[&str] = &[
    "idiot", "stupid", "moron", "dumbass", "dumb", "loser", "trash", "garbage",
    "scumbag", "shutup", "hateyou", "killyourself", "kys", "worthless", "pathetic",
    "freak", "noob", "fuck", "fucking", "shit", "asshole", "bastard", "bitch",
    "dick", "wtf", "stfu",
];

/// How many seed words a message hits (folded matching, same rules as the
/// censor: whole token or spans a separator).
fn risk_lexicon_hits(text: &str) -> usize {
    let c = condense(text);
    let chars: Vec<char> = c.text.chars().collect();
    let n = chars.len();
    let mut hits = 0;
    for word in RISK_SEED {
        let wchars: Vec<char> = condense(word).text.chars().collect();
        let m = wchars.len();
        if m == 0 || m > n {
            continue;
        }
        for a in 0..=n - m {
            if chars[a..a + m] == wchars[..] && is_hit(text, &c, a, m) {
                hits += 1;
                break;
            }
        }
    }
    hits
}

/// The interpretable signals the model regresses on.
struct RiskFeatures {
    /// Saturating lexicon signal: 0 with no seed hit, approaching 1 with several.
    lexicon: f32,
    /// Obfuscation density: separators tucked inside a word + leet-folded chars.
    obfuscation: f32,
    /// Shouting: uppercase share of letters.
    caps: f32,
    /// Stretch: share of characters sitting in a run of >= 3 identical chars.
    repetition: f32,
    /// Symbol dust: share of non-alphanumeric, non-space characters.
    symbols: f32,
    /// Length, saturating at ~200 chars.
    length: f32,
}

fn risk_features(text: &str) -> RiskFeatures {
    let total = text.chars().count().max(1);
    let letters = text.chars().filter(|c| c.is_alphabetic()).count();
    let upper = text.chars().filter(|c| c.is_uppercase()).count();
    let symbols = text
        .chars()
        .filter(|c| !c.is_alphanumeric() && !c.is_whitespace())
        .count();
    let leet = text
        .chars()
        .filter(|c| fold(*c) != c.to_ascii_lowercase())
        .count();

    // A separator tucked between letters of the SAME run ("b.a.d") is the
    // obfuscation we care about; whitespace ends the run (a normal word gap).
    let mut tucked = 0usize;
    let mut in_run = false;
    for ch in text.chars() {
        if ch.is_alphabetic() {
            in_run = true;
        } else if ch.is_whitespace() {
            in_run = false;
        } else if in_run {
            tucked += 1;
        }
    }

    let mut repetition = 0usize;
    let mut prev = '\0';
    let mut run = 0usize;
    for ch in text.chars() {
        if ch == prev {
            run += 1;
        } else {
            prev = ch;
            run = 1;
        }
        if run >= 3 {
            repetition += 1;
        }
    }

    RiskFeatures {
        lexicon: 1.0 - (-(risk_lexicon_hits(text) as f32)).exp(),
        obfuscation: ((tucked + leet) as f32 / total as f32).min(1.0),
        caps: if letters > 0 {
            upper as f32 / letters as f32
        } else {
            0.0
        },
        repetition: (repetition as f32 / total as f32).min(1.0),
        symbols: (symbols as f32 / total as f32).min(1.0),
        length: (total as f32 / 200.0).min(1.0),
    }
}

// Logistic weights (hand-tuned; the model is a plain linear classifier).
const RISK_W_LEXICON: f32 = 6.0;
const RISK_W_OBFUSCATION: f32 = 3.0;
const RISK_W_CAPS: f32 = 0.5;
const RISK_W_REPETITION: f32 = 2.0;
const RISK_W_SYMBOLS: f32 = 1.0;
const RISK_W_LENGTH: f32 = 0.5;
const RISK_BIAS: f32 = -3.0;

/// Score a message 0..1 for "risk". Fast, local, and dependency-free.
fn risk_score(text: &str) -> f32 {
    let f = risk_features(text);
    let z = RISK_BIAS
        + RISK_W_LEXICON * f.lexicon
        + RISK_W_OBFUSCATION * f.obfuscation
        + RISK_W_CAPS * f.caps
        + RISK_W_REPETITION * f.repetition
        + RISK_W_SYMBOLS * f.symbols
        + RISK_W_LENGTH * f.length;
    1.0 / (1.0 + (-z).exp())
}

/// Optional LLM review backend. Spawns `worker/review_worker.py` lazily (only
/// when enabled), feeds it JSONL requests on stdin, reads the 0–1 risk back.
/// A review that times out (slow model/load) yields None and the message is
/// passed through unreviewed — the module never blocks the pipeline ack.
struct ReviewWorker {
    enabled: bool,
    engine: String,
    threshold: f32,
    review_timeout_secs: u32,
    python_interpreter: String,
    worker_script_path: String,
    llama_model: String,
    deberta_model: String,
    deberta_max_length: u32,
    llama_max_length: u32,
    llama_max_new_tokens: u32,
    child: AsyncMutex<Option<ReviewChild>>,
    dead: Arc<AtomicBool>,
}

struct ReviewChild {
    /// The spawned process handle — kept so every error path can kill + reap it
    /// instead of leaking an orphaned Python worker across restarts.
    child: tokio::process::Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ReviewWorker {
    /// Settings (timeouts, interpreter, worker path, model overrides) are read
    /// from the module Config so operators can tune them without editing code.
    fn new(enabled: bool, engine: String, threshold: f32, cfg: &Config) -> Self {
        Self {
            enabled,
            engine,
            threshold,
            review_timeout_secs: cfg.review_timeout_secs,
            python_interpreter: cfg.python_interpreter.clone(),
            worker_script_path: cfg.worker_script_path.clone(),
            llama_model: cfg.llama_model.clone(),
            deberta_model: cfg.deberta_model.clone(),
            deberta_max_length: cfg.deberta_max_length,
            llama_max_length: cfg.llama_max_length,
            llama_max_new_tokens: cfg.llama_max_new_tokens,
            child: AsyncMutex::new(None),
            dead: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Re-enable review after a reconnect (fresh connection = fresh start).
    fn reset_dead(&self) {
        self.dead.store(false, Ordering::SeqCst);
    }

    /// Score a message. None = disabled, worker failed to start, timed out, or
    /// an error — the caller passes the message through unreviewed.
    ///
    /// The whole request-write + response-read round trip runs under a single
    /// timeout so a hung worker (e.g. llama-guard on CPU) can never wedge the
    /// caller. Each request carries an `id` that the worker echoes back; any
    /// response whose id doesn't match (a stale reply from a previously
    /// abandoned request) is discarded, so a risk can never be mis-attributed
    /// to the wrong message. On timeout or worker error the child is killed and
    /// reaped; the next review spawns a fresh worker (reviving `dead`).
    async fn review(&self, text: &str) -> Option<f32> {
        if !self.enabled {
            return None;
        }
        let mut guard = self.child.lock().await;
        if guard.is_none() {
            match spawn_review_worker(self).await {
                Ok(c) => {
                    // A transient failure set `dead`; a successful spawn revives
                    // review instead of leaving it disabled forever.
                    self.dead.store(false, Ordering::SeqCst);
                    *guard = Some(c);
                }
                Err(e) => {
                    warn!(
                        "llm_review: worker failed to start ({}); disabling review",
                        e
                    );
                    self.dead.store(true, Ordering::SeqCst);
                    return None;
                }
            }
        }
        let child = guard.as_mut()?;
        let id = uuid::Uuid::now_v7().to_string();
        let req = serde_json::json!({ "id": id, "text": text }).to_string();

        let round_trip = async {
            if child.stdin.write_all(req.as_bytes()).await.is_err()
                || child.stdin.write_all(b"\n").await.is_err()
            {
                return Err("write to worker failed".to_string());
            }
            loop {
                let mut line = String::new();
                match child.stdout.read_line(&mut line).await {
                    Ok(0) => return Err("worker closed stdout".to_string()),
                    Ok(_) => {}
                    Err(e) => return Err(format!("read from worker failed: {}", e)),
                }
                let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) else {
                    continue; // partial/garbage line — keep reading
                };
                // Discard stale responses (from a previously timed-out request)
                // so they can't be attributed to the current message.
                if v.get("id").and_then(|i| i.as_str()) != Some(id.as_str()) {
                    continue;
                }
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(format!("worker error: {}", err));
                }
                return Ok(v.get("risk").and_then(|r| r.as_f64()).map(|f| f as f32));
            }
        };

        match tokio::time::timeout(
            Duration::from_secs(self.review_timeout_secs as u64),
            round_trip,
        )
        .await
        {
            Ok(Ok(Some(risk))) => Some(risk),
            Ok(Ok(None)) => None, // worker responded without a usable risk field
            Ok(Err(e)) => {
                // The worker is broken (model missing / HF-gated / died): warn +
                // disable so we stop paying per-message timeouts, kill + reap it.
                warn!("llm_review worker error ({}); killing worker", e);
                self.dead.store(true, Ordering::SeqCst);
                if let Some(c) = guard.take() {
                    kill_and_reap(c).await;
                }
                None
            }
            Err(_) => {
                // Round-trip timed out — the worker may still be churning on our
                // request. Kill + reap so its eventual response can't poison the
                // next review; a fresh worker respawns on the next review.
                warn!("llm_review round-trip timed out; killing worker");
                self.dead.store(true, Ordering::SeqCst);
                if let Some(c) = guard.take() {
                    kill_and_reap(c).await;
                }
                None
            }
        }
    }

    /// Whether this text should be treated as a hit given the configurable
    /// risk-certainty threshold.
    fn is_hit(&self, risk: f32) -> bool {
        risk >= self.threshold
    }
}

/// Kill + reap a worker. Called on every error/timeout path so no orphaned
/// Python processes accumulate across restarts.
async fn kill_and_reap(mut child: ReviewChild) {
    let _ = child.child.kill().await;
    let _ = child.child.wait().await;
}

impl Drop for ReviewChild {
    fn drop(&mut self) {
        // Module-exit safety net: best-effort kill so a live worker can't
        // outlive the module. (The async error paths kill + wait explicitly.)
        let _ = self.child.start_kill();
    }
}

async fn spawn_review_worker(
    worker: &ReviewWorker,
) -> Result<ReviewChild, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(&worker.python_interpreter);
    cmd.arg(&worker.worker_script_path)
        .arg("--engine")
        .arg(&worker.engine)
        .arg("--llama-model")
        .arg(&worker.llama_model)
        .arg("--deberta-model")
        .arg(&worker.deberta_model)
        .arg("--deberta-max-length")
        .arg(worker.deberta_max_length.to_string())
        .arg("--llama-max-length")
        .arg(worker.llama_max_length.to_string())
        .arg("--llama-max-new-tokens")
        .arg(worker.llama_max_new_tokens.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    let mut child = cmd.spawn()?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    Ok(ReviewChild {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

/// Build the ChatMessageRejected record a module sends to the engine so the
/// rejection is logged clearly (reason + original raw + processed).
fn compose_rejected(
    message_uuid7: &str,
    chat: &ChatMessage,
    processed: &str,
    reason: &str,
) -> ChatMessageRejected {
    ChatMessageRejected {
        message_uuid7: message_uuid7.to_string(),
        message: Some(chat.clone()),
        processed_message: Some(processed.to_string()),
        reason: reason.to_string(),
        origin: "banned-words".to_string(),
    }
}

/// An empty command list registers this module as a catch-all, so the engine
/// routes every message (including command messages like `!tts <text>`) here.
fn catch_all_commands() -> Commands {
    Commands {
        commands: vec![],
        alert_on_unknown_command: false,
    }
}

async fn send_container(
    write_shared: &Arc<AsyncMutex<WsWriteHalf>>,
    container: ContainerForEngine,
) {
    let mut buf = Vec::new();
    if container.encode(&mut buf).is_ok() {
        let mut w = write_shared.lock().await;
        let _ = w.send(WsMessage::Binary(buf)).await;
    }
}

/// Identity the engine assigned to this module session (refreshed on reconnect).
struct EngineIdentity {
    auth: String,
    module: String,
    instance: String,
}

/// Handle a single MessageInProcess: run the word-list detector (and, when
/// enabled, the LLM review), censor if flagged, and reply with the processed
/// message so the engine acks in_process. If `flag_for_review`, also log a
/// ChatMessageRejected. Called either inline from the read loop (word-list
/// path) or from a spawned task (LLM-review path, so the loop keeps answering
/// AuthVerify while the worker runs).
async fn process_message(
    config: &Config,
    worker: &Option<Arc<ReviewWorker>>,
    chat: &ChatMessage,
    incoming: &str,
    uuid: &str,
    identity: &EngineIdentity,
    write_shared: &Arc<AsyncMutex<WsWriteHalf>>,
) {
    // Operate on the accumulated in-process text from earlier modules, not the
    // raw message: censoring the raw message would revert their edits.
    let original = effective_text(&chat.raw_message, incoming);
    let mut flagged = false;
    let mut reason = String::new();
    match detect_banned(original, &config.banned_words) {
        Some((word, variant)) => {
            warn!("Banned word '{}' detected via {} variant", word, variant);
            flagged = true;
            reason = format!("banned word '{}' via {} variant", word, variant);
        }
        None => {
            // Lightweight in-process AI risk model: fast, local, no Python or
            // model files. A risk >= the configurable threshold is a hit.
            if config.risk_model {
                let risk = risk_score(original);
                if risk >= config.risk_threshold {
                    warn!("Risk model flagged message (risk={:.3})", risk);
                    flagged = true;
                    reason = format!("risk {:.3} >= threshold {}", risk, config.risk_threshold);
                }
            }
            // Optional heavier LLM review (Python worker): catch what the word
            // list AND the risk model missed. Only if not already flagged.
            if !flagged && config.llm_review {
                if let Some(w) = worker {
                    if let Some(risk) = w.review(original).await {
                        if w.is_hit(risk) {
                            warn!("LLM review flagged message (risk={})", risk);
                            flagged = true;
                            reason = format!(
                                "LLM risk {} >= threshold {}",
                                risk, config.llm_review_threshold
                            );
                        }
                    }
                }
            }
        }
    }

    let censored = if flagged {
        apply_censor(config, original)
    } else {
        original.to_string()
    };

    // If flag_for_review, send a ChatMessageRejected so the engine logs the
    // rejection clearly (reason + raw + processed) instead of an obscure log.
    if flagged && config.flag_for_review {
        let rej = compose_rejected(uuid, chat, &censored, &reason);
        let log = ContainerForEngine {
            version: 2,
            auth_token: identity.auth.clone(),
            module_name: identity.module.clone(),
            module_instance_uuid7: identity.instance.clone(),
            payload: Some(EnginePayload::ChatMessageRejected(rej)),
        };
        let mut buf = Vec::new();
        if log.encode(&mut buf).is_ok() {
            let mut w = write_shared.lock().await;
            let _ = w.send(WsMessage::Binary(buf)).await;
        }
    }

    // Reply with the (possibly censored) message, same uuid → engine acks in_process.
    let reply = ContainerForEngine {
        version: 2,
        auth_token: identity.auth.clone(),
        module_name: identity.module.clone(),
        module_instance_uuid7: identity.instance.clone(),
        payload: Some(EnginePayload::MessageInProcess(MessageInProcess {
            message_uuid7: uuid.to_string(),
            raw_message: Some(ChatMessage {
                platform: chat.platform.clone(),
                raw_data: chat.raw_data.clone(),
                raw_message: chat.raw_message.clone(),
                user_uuid7: chat.user_uuid7.clone(),
                command: chat.command.clone(),
                channel_id: chat.channel_id.clone(),
                user_data: chat.user_data.clone(),
            }),
            processed_message: censored.clone(),
            abandon_message: false,
            audio: Vec::new(),
            audio_type: String::new(),
        })),
    };
    let mut buf = Vec::new();
    if reply.encode(&mut buf).is_ok() {
        let mut w = write_shared.lock().await;
        let _ = w.send(WsMessage::Binary(buf)).await;
        if flagged {
            info!(
                "Censored message ({}): {:?} -> {:?}",
                uuid, original, censored
            );
        }
    }
}

/// Apply the configured censor (sentence or token mode) to a flagged message.
fn apply_censor(config: &Config, original: &str) -> String {
    match config.sentence_mode.as_str() {
        // Censor the whole message/sentence.
        "censor" => "*".repeat(original.chars().count()),
        // Replace the whole message/sentence.
        "replace" => {
            if config.replace_sentence.is_empty() {
                censor_message(
                    original,
                    &config.banned_words,
                    &config.censor_mode,
                    &config.replace_word,
                )
            } else {
                config.replace_sentence.clone()
            }
        }
        // Default: censor only the offending token(s).
        _ => censor_message(
            original,
            &config.banned_words,
            &config.censor_mode,
            &config.replace_word,
        ),
    }
}

/// Send a Prompt to the engine (forwarded to connected UIs) and wait for the
/// operator's response (`PromptResponse.reason`). Returns None on cancel/timeout.
async fn prompt_for_input(
    write_ws: &Arc<AsyncMutex<WsWriteHalf>>,
    prompt_rx: &mut mpsc::UnboundedReceiver<PromptResponse>,
    auth_token: &str,
    module_name: &str,
    instance_uuid: &str,
    title: &str,
    details: &str,
    input_label: &str,
    kind: PromptKind,
    timeout: u32,
) -> Option<String> {
    let prompt_id = uuid::Uuid::now_v7().to_string();
    let prompt_type = match kind {
        PromptKind::Boolean => PromptType::Boolean,
        PromptKind::String => PromptType::String,
        PromptKind::Credential => PromptType::Credential,
    };
    let prompt = Prompt {
        prompt_id_uuid7: prompt_id.clone(),
        prompt: title.to_string(),
        details: details.to_string(),
        yes_dialog: "Submit".to_string(),
        no_dialog: "Cancel".to_string(),
        timeout,
        origin: module_name.to_string(),
        origin_uuid7: String::new(),
        instructions: String::new(),
        link: String::new(),
        input_label: input_label.to_string(),
        prompt_type: prompt_type as i32,
    };
    let container = ContainerForEngine {
        version: 2,
        auth_token: auth_token.to_string(),
        module_name: module_name.to_string(),
        module_instance_uuid7: instance_uuid.to_string(),
        payload: Some(EnginePayload::Prompt(prompt)),
    };
    let mut buf = Vec::new();
    if container.encode(&mut buf).is_err() {
        return None;
    }
    let mut guard = write_ws.lock().await;
    if guard.send(WsMessage::Binary(buf)).await.is_err() {
        return None;
    }
    drop(guard);

    let deadline = tokio::time::Instant::now() + Duration::from_secs((timeout + 10) as u64);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(10), prompt_rx.recv()).await {
            Ok(Some(resp)) if resp.prompt_id_uuid7 == prompt_id => {
                return if resp.accepted {
                    Some(resp.reason)
                } else {
                    None
                };
            }
            Ok(Some(_)) => continue, // a different prompt's response
            Ok(None) => return None,
            // The 10s poll interval elapsed with no response yet: keep waiting
            // until the real deadline (the `timeout` seconds above), rather than
            // bailing out 10 seconds in and auto-cancelling every prompt.
            Err(_) => continue,
        }
    }
    None
}

/// Read the Config from config.json. Prefers the `module_specific` object
/// (where the engine/TUI store credential settings); falls back to the legacy
/// top-level layout so pre-existing configs keep working.
fn read_config_from_file() -> Option<Config> {
    let s = std::fs::read_to_string("config.json").ok()?;
    let root: serde_json::Value = serde_json::from_str(&s).ok()?;
    match root.get("module_specific") {
        Some(ms) if ms.is_object() => serde_json::from_value::<Config>(ms.clone()).ok(),
        _ => serde_json::from_str::<Config>(&s).ok(),
    }
}

/// Save the Config into the `module_specific` object of config.json, merging
/// with (and preserving) any existing top-level fields such as ip/port/pin.
fn save_config(config: &Config) {
    let mut root: serde_json::Value = std::fs::read_to_string("config.json")
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .unwrap_or_else(|| serde_json::json!({}));
    root["module_specific"] = serde_json::to_value(config).unwrap_or(serde_json::Value::Null);
    if let Ok(pretty) = serde_json::to_string_pretty(&root) {
        let _ = std::fs::write("config.json", pretty);
    }
}

fn default_config() -> Config {
    Config {
        banned_words: vec!["cheese".to_string(), "badword".to_string()],
        censor_mode: default_mode(),
        replace_word: String::new(),
        flag_for_review: false,
        sentence_mode: default_sentence_mode(),
        replace_sentence: String::new(),
        llm_review: false,
        llm_review_threshold: default_review_threshold(),
        llm_review_engine: default_review_engine(),
        review_timeout_secs: default_review_timeout_secs(),
        python_interpreter: default_python_interpreter(),
        worker_script_path: default_worker_script_path(),
        prompt_timeout_secs: default_prompt_timeout_secs(),
        reconnect_base_secs: default_reconnect_base_secs(),
        reconnect_max_secs: default_reconnect_max_secs(),
        llama_model: default_llama_model(),
        deberta_model: default_deberta_model(),
        deberta_max_length: default_deberta_max_length(),
        llama_max_length: default_llama_max_length(),
        llama_max_new_tokens: default_llama_max_new_tokens(),
        risk_model: false,
        risk_threshold: default_risk_threshold(),
    }
}

async fn load_config(
    write_ws: &Arc<AsyncMutex<WsWriteHalf>>,
    prompt_rx: &mut mpsc::UnboundedReceiver<PromptResponse>,
    auth_token: &str,
    module_name: &str,
    instance_uuid: &str,
) -> Config {
    if let Some(cfg) = read_config_from_file() {
        // A legacy top-level config is migrated into module_specific here so
        // the engine's stored credentials and this module stay in sync.
        save_config(&cfg);
        return cfg;
    }

    // No saved config: ask the operator via the engine prompt subwindow
    // instead of silently writing a default.
    let banned_answer = prompt_for_input(
        write_ws,
        prompt_rx,
        auth_token,
        module_name,
        instance_uuid,
        "Configure banned-words module",
        "No saved config was found. Enter the list of words to filter, separated by commas.",
        "Banned words (comma-separated)",
        PromptKind::String,
        default_config().prompt_timeout_secs,
    )
    .await;

    let mode_answer = prompt_for_input(
        write_ws,
        prompt_rx,
        auth_token,
        module_name,
        instance_uuid,
        "Configure banned-words module",
        "Enter the censor mode used when a banned word is detected.",
        "censor mode: soft|mid|hard|replace",
        PromptKind::String,
        default_config().prompt_timeout_secs,
    )
    .await;

    let default = default_config();
    let config = Config {
        banned_words: banned_answer
            .as_deref()
            .map(|s| {
                s.split(',')
                    .map(|w| w.trim().to_string())
                    .filter(|w| !w.is_empty())
                    .collect::<Vec<String>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or(default.banned_words),
        censor_mode: mode_answer
            .as_deref()
            .filter(|m| matches!(*m, "soft" | "mid" | "hard" | "replace"))
            .map(|m| m.to_string())
            .unwrap_or(default.censor_mode),
        replace_word: String::new(),
        flag_for_review: false,
        sentence_mode: default_sentence_mode(),
        replace_sentence: String::new(),
        // LLM review defaults off; the operator enables it + sets the
        // risk-certainty threshold via the module's config (engine/TUI).
        llm_review: default.llm_review,
        llm_review_threshold: default.llm_review_threshold,
        llm_review_engine: default.llm_review_engine,
        review_timeout_secs: default.review_timeout_secs,
        python_interpreter: default.python_interpreter,
        worker_script_path: default.worker_script_path,
        prompt_timeout_secs: default.prompt_timeout_secs,
        reconnect_base_secs: default.reconnect_base_secs,
        reconnect_max_secs: default.reconnect_max_secs,
        llama_model: default.llama_model,
        deberta_model: default.deberta_model,
        deberta_max_length: default.deberta_max_length,
        llama_max_length: default.llama_max_length,
        llama_max_new_tokens: default.llama_max_new_tokens,
        risk_model: default.risk_model,
        risk_threshold: default.risk_threshold,
    };

    save_config(&config);
    config
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(tracing::Level::INFO)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::set_global_default(subscriber).unwrap();

    // Connect to the engine as an in-process module (CLI overrides: --ip/--port/--pin).
    let client = CockatielClient::connect("banned_words.json").await?;
    let (write, read) = client.stream.split();
    let write_shared: Arc<AsyncMutex<WsWriteHalf>> = Arc::new(AsyncMutex::new(write));
    let auth_token = client.auth_token.clone();
    let instance_uuid = client.instance_uuid7.clone();
    let module_name = client.config.module_name.clone();

    // Channel carrying PromptResponses from the engine to the config prompt,
    // so `prompt_for_input` can await the operator's typed answer.
    let (prompt_tx, mut prompt_rx) = mpsc::unbounded_channel::<PromptResponse>();

    // Shared config: the read task must run while config is being loaded (so
    // the config prompt can receive its response), so it reads config from an
    // Arc<Mutex> that is populated right after load_config returns.
    let config_shared: Arc<Mutex<Config>> = Arc::new(Mutex::new(default_config()));
    // Shared optional LLM reviewer — populated after load_config; the read task
    // clones it per message so a disabled review never spawns anything.
    let review_worker_shared: Arc<Mutex<Option<Arc<ReviewWorker>>>> = Arc::new(Mutex::new(None));

    // Read task: forward PromptResponses to the awaiting prompt AND handle
    // message in-processing. Spawned BEFORE load_config so prompts work.
    // Owns the read half + identity so it can reconnect with backoff when the
    // engine drops the socket (instead of dying and leaving main to sleep
    // forever while the watchdog severs the unresponsive module).
    {
        let prompt_tx_task = prompt_tx.clone();
        let config_shared = Arc::clone(&config_shared);
        let review_worker_shared = Arc::clone(&review_worker_shared);
        let write_shared = Arc::clone(&write_shared);
        // Identity + read half live in the task so a reconnect can refresh them
        // (the engine forgets a session when the socket drops).
        let mut auth_token = auth_token.clone();
        let mut instance_uuid = instance_uuid.clone();
        let mut module_name = module_name.clone();
        let mut read = read;
        tokio::spawn(async move {
            'reconnect: loop {
                // Register as a catch-all (empty command list) on every
                // connect so command messages (e.g. `!tts <slur>`) reach the
                // censor too, not just the owning command module.
                send_container(
                    &write_shared,
                    ContainerForEngine {
                        version: 2,
                        auth_token: auth_token.clone(),
                        module_name: module_name.clone(),
                        module_instance_uuid7: instance_uuid.clone(),
                        payload: Some(EnginePayload::Commands(catch_all_commands())),
                    },
                )
                .await;
                loop {
                    let Some(msg) = read.next().await else { break };
                    let data = match msg {
                        Ok(WsMessage::Binary(d)) => d,
                        Ok(WsMessage::Close(_)) => {
                            info!("Engine closed connection");
                            break;
                        }
                        Ok(_) => continue,
                        Err(e) => {
                            warn!("Engine WebSocket error: {}", e);
                            break;
                        }
                    };
                    let Ok(container) = ContainerForModule::decode(data.as_ref()) else {
                        continue;
                    };

                    match container.payload {
                        Some(ModulePayload::AuthVerify(_)) => {
                            // Answer the engine's liveness probe (this module
                            // reads the socket directly, so the client's
                            // auto-answer is bypassed — without this the
                            // watchdog severs us).
                            let reply = ContainerForEngine {
                                version: 2,
                                auth_token: auth_token.clone(),
                                module_name: module_name.clone(),
                                module_instance_uuid7: instance_uuid.clone(),
                                payload: Some(EnginePayload::AuthVerify(AuthVerify {
                                    cur_auth: auth_token.clone(),
                                })),
                            };
                            let mut buf = Vec::new();
                            if reply.encode(&mut buf).is_ok() {
                                let mut w = write_shared.lock().await;
                                let _ = w.send(WsMessage::Binary(buf)).await;
                            }
                        }
                        Some(ModulePayload::PromptResponse(resp)) => {
                            // Forward operator answers to the awaiting prompt.
                            let _ = prompt_tx_task.send(resp);
                        }
                        Some(ModulePayload::MessageInProcess(process)) => {
                            let Some(chat) = &process.raw_message else {
                                continue;
                            };
                            let config = config_shared.lock().unwrap().clone();
                            // The accumulated in-process text from earlier
                            // modules (e.g. language-constrainer); fall back to
                            // the raw message only when it is empty.
                            let incoming = process.processed_message.clone();
                            let text = effective_text(&chat.raw_message, &incoming);
                            let uuid = process.message_uuid7.clone();
                            if !uuid.is_empty() {
                                let receipt = ContainerForEngine {
                                    version: 2,
                                    auth_token: auth_token.clone(),
                                    module_name: module_name.clone(),
                                    module_instance_uuid7: instance_uuid.clone(),
                                    payload: Some(EnginePayload::MessageAck(MessageAck {
                                        message_uuid7: uuid.clone(),
                                    })),
                                };
                                let mut buf = Vec::new();
                                if receipt.encode(&mut buf).is_ok() {
                                    let mut w = write_shared.lock().await;
                                    let _ = w.send(WsMessage::Binary(buf)).await;
                                }
                            }
                            let worker = review_worker_shared.lock().unwrap().clone();

                            // LLM review can take up to ~10s and may time out.
                            // Run it OFF the read loop (spawned task) so the
                            // loop keeps answering AuthVerify probes while the
                            // worker runs. Only detour when the word list
                            // missed AND review is enabled — word hits keep the
                            // fast inline path.
                            if config.llm_review
                                && worker.is_some()
                                && detect_banned(text, &config.banned_words).is_none()
                            {
                                let identity = EngineIdentity {
                                    auth: auth_token.clone(),
                                    module: module_name.clone(),
                                    instance: instance_uuid.clone(),
                                };
                                let (cfg, w, ch, inc, u, idnt, ws) = (
                                    config.clone(),
                                    worker.clone().unwrap(),
                                    chat.clone(),
                                    incoming.clone(),
                                    uuid.clone(),
                                    identity,
                                    Arc::clone(&write_shared),
                                );
                                tokio::spawn(async move {
                                    process_message(&cfg, &Some(w), &ch, &inc, &u, &idnt, &ws)
                                        .await;
                                });
                                continue;
                            }

                            let identity = EngineIdentity {
                                auth: auth_token.clone(),
                                module: module_name.clone(),
                                instance: instance_uuid.clone(),
                            };
                            process_message(
                                &config,
                                &worker,
                                chat,
                                &incoming,
                                &uuid,
                                &identity,
                                &write_shared,
                            )
                            .await;
                        }
                        Some(ModulePayload::Shutdown(_)) => {
                            info!("Engine requested shutdown — exiting");
                            std::process::exit(0);
                        }
                        _ => {}
                    }
                }

                // The engine connection dropped — reconnect with backoff
                // instead of leaving the module unresponsive.
                info!("Engine disconnected — reconnecting...");
                let reconnect_cfg = config_shared.lock().unwrap().clone();
                let mut backoff = reconnect_cfg.reconnect_base_secs;
                loop {
                    tokio::time::sleep(Duration::from_secs(backoff as u64)).await;
                    match CockatielClient::connect("banned_words.json").await {
                        Ok(conn) => {
                            info!("Reconnected to engine");
                            let (w, r) = conn.stream.split();
                            *write_shared.lock().await = w;
                            auth_token = conn.auth_token;
                            instance_uuid = conn.instance_uuid7;
                            module_name = conn.config.module_name;
                            read = r;
                            // A fresh session is a fresh start: a transient
                            // worker failure must not persist across a reconnect.
                            if let Some(w) = review_worker_shared.lock().unwrap().clone() {
                                w.reset_dead();
                            }
                            continue 'reconnect;
                        }
                        Err(e) => {
                            warn!("Engine reconnect failed: {} — retrying in {}s", e, backoff);
                            backoff = (backoff * 2).min(reconnect_cfg.reconnect_max_secs);
                        }
                    }
                }
            }
        });
    }

    let config = load_config(
        &write_shared,
        &mut prompt_rx,
        &auth_token,
        &module_name,
        &instance_uuid,
    )
    .await;
    *config_shared.lock().unwrap() = config.clone();
    *review_worker_shared.lock().unwrap() = Some(Arc::new(ReviewWorker::new(
        config.llm_review,
        config.llm_review_engine.clone(),
        config.llm_review_threshold,
        &config,
    )));
    info!(
        "Banned-words module active: mode={}, {} word(s), llm_review={} (engine={}, threshold={})",
        config.censor_mode,
        config.banned_words.len(),
        config.llm_review,
        config.llm_review_engine,
        config.llm_review_threshold,
    );

    // Keep the process alive; the read task does all the work.
    loop {
        tokio::time::sleep(Duration::from_secs(3600)).await;
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    fn banned() -> Vec<String> {
        vec!["badword".to_string(), "shoot".to_string()]
    }

    #[test]
    fn catch_all_commands_is_empty_and_quiet() {
        let c = catch_all_commands();
        assert!(c.commands.is_empty());
        assert!(!c.alert_on_unknown_command);
    }

    #[test]
    fn detects_exact_match() {
        let (w, variant) = detect_banned("you are a badword", &banned()).unwrap();
        assert_eq!(w, "badword");
        assert_eq!(variant, "exact");
    }

    #[test]
    fn detects_leet() {
        assert!(detect_banned("b4dw0rd", &banned()).is_some());
    }

    #[test]
    fn detects_no_spaces() {
        assert!(detect_banned("b a d w o r d", &banned()).is_some());
        assert!(detect_banned("b_a_d_w_o_r_d", &banned()).is_some());
    }

    #[test]
    fn case_insensitive() {
        let (w, _) = detect_banned("BADWORD", &banned()).unwrap();
        assert_eq!(w, "badword");
    }

    #[test]
    fn clean_message_is_none() {
        assert!(detect_banned("hello friends", &banned()).is_none());
    }

    #[test]
    fn substring_of_a_larger_word_is_not_flagged() {
        // A banned "ass" must not flag "class"/"grass", nor "hell" flag "hello".
        let list = vec!["ass".to_string(), "hell".to_string()];
        assert!(detect_banned("class", &list).is_none());
        assert!(detect_banned("grass", &list).is_none());
        assert!(detect_banned("hello", &list).is_none());
        assert!(detect_banned("a classic grassy hello", &list).is_none());
    }

    #[test]
    fn standalone_and_leet_words_are_still_flagged() {
        let list = vec!["ass".to_string(), "hell".to_string()];
        assert!(detect_banned("ass", &list).is_some());
        assert!(detect_banned("you ass", &list).is_some());
        assert!(detect_banned("h3ll", &list).is_some());
        assert!(detect_banned("go to h3ll", &list).is_some());
    }

    #[test]
    fn substring_censor_does_not_touch_larger_words() {
        let list = vec!["ass".to_string()];
        assert_eq!(censor_message("class", &list, "hard", ""), "class");
        assert_eq!(
            censor_message("grass is ass", &list, "hard", ""),
            "grass is ***"
        );
    }

    #[test]
    fn prior_in_process_edit_is_preserved_and_censored() {
        // A prior module (language-constrainer) rewrote the message. banned_words
        // must censor THAT text, not revert to the raw message.
        let mut cfg = default_config();
        cfg.banned_words = vec!["badword".to_string()];
        cfg.censor_mode = "hard".to_string();
        let incoming = "constrainer cleaned badword here";
        let text = effective_text("raw original badword text", incoming);
        assert_eq!(text, incoming, "the earlier edit must be preserved");
        assert!(detect_banned(text, &cfg.banned_words).is_some());
        assert_eq!(apply_censor(&cfg, text), "constrainer cleaned ******* here");
    }

    #[test]
    fn empty_incoming_falls_back_to_raw() {
        let mut cfg = default_config();
        cfg.banned_words = vec!["badword".to_string()];
        cfg.censor_mode = "hard".to_string();
        assert_eq!(effective_text("raw badword", ""), "raw badword");
        assert_eq!(
            apply_censor(&cfg, effective_text("raw badword", "")),
            "raw *******"
        );
    }

    #[test]
    fn censor_hard_stars_everything() {
        assert_eq!(censor_token("badword", "hard", ""), "*******");
    }

    #[test]
    fn censor_soft_keeps_ends() {
        assert_eq!(censor_token("badword", "soft", ""), "b*****d");
        assert_eq!(censor_token("sh", "soft", ""), "**");
    }

    #[test]
    fn censor_replace_uses_replace_word() {
        assert_eq!(censor_token("badword", "replace", "heck"), "heck");
        assert_eq!(censor_token("badword", "replace", ""), "*******");
    }

    #[test]
    fn censor_message_censors_matching_tokens_only() {
        let out = censor_message("badword is a word", &banned(), "hard", "");
        assert_eq!(out, "******* is a word");
    }

    #[test]
    fn censor_message_multibyte_whitespace_delimiter_no_panic() {
        // U+3000 (ideographic space) and U+00A0 (no-break space) are accepted
        // by char::is_whitespace — byte-slicing them used to panic mid-char.
        assert_eq!(
            censor_message("badword\u{3000}ok", &banned(), "hard", ""),
            "*******\u{3000}ok"
        );
        assert_eq!(
            censor_message("badword\u{00a0}ok", &banned(), "hard", ""),
            "*******\u{00a0}ok"
        );
        // Trailing multi-byte whitespace must also survive.
        assert_eq!(
            censor_message("badword\u{3000}", &banned(), "hard", ""),
            "*******\u{3000}"
        );
    }

    fn cfg_with_review() -> Config {
        Config {
            llm_review: true,
            llm_review_threshold: 0.5,
            llm_review_engine: "deberta".to_string(),
            ..default_config()
        }
    }

    fn test_worker(enabled: bool, engine: &str, threshold: f32) -> ReviewWorker {
        ReviewWorker::new(enabled, engine.to_string(), threshold, &default_config())
    }

    #[test]
    fn llm_review_off_by_default() {
        assert!(!default_config().llm_review);
        assert_eq!(default_config().llm_review_threshold, 0.5);
        assert_eq!(default_config().llm_review_engine, "deberta");
    }

    #[test]
    fn new_tuning_defaults_and_partial_deserialize() {
        let d = default_config();
        assert_eq!(d.review_timeout_secs, 10);
        assert_eq!(d.python_interpreter, "python3");
        assert_eq!(d.worker_script_path, "worker/review_worker.py");
        assert_eq!(d.prompt_timeout_secs, 60);
        assert_eq!(d.reconnect_base_secs, 1);
        assert_eq!(d.reconnect_max_secs, 30);
        assert_eq!(d.llama_model, "meta-llama/Llama-Guard-3-1B");
        assert_eq!(d.deberta_model, "microsoft/deberta-v3-small");
        assert_eq!(d.deberta_max_length, 256);
        assert_eq!(d.llama_max_length, 2000);
        assert_eq!(d.llama_max_new_tokens, 16);
        // A legacy/partial module_specific object still gets every new default.
        let partial: Config = serde_json::from_value(serde_json::json!({
            "banned_words": ["x"],
            "censor_mode": "hard",
        }))
        .unwrap();
        assert_eq!(partial.review_timeout_secs, 10);
        assert_eq!(partial.llama_model, "meta-llama/Llama-Guard-3-1B");
        assert_eq!(partial.deberta_max_length, 256);
        assert_eq!(partial.llama_max_new_tokens, 16);
        // An explicit override is honored.
        let overridden: Config = serde_json::from_value(serde_json::json!({
            "llama_max_new_tokens": 64,
            "review_timeout_secs": 20,
        }))
        .unwrap();
        assert_eq!(overridden.llama_max_new_tokens, 64);
        assert_eq!(overridden.review_timeout_secs, 20);
    }

    #[test]
    fn threshold_hit_and_miss() {
        let w = test_worker(true, "deberta", 0.5);
        assert!(w.is_hit(0.9));
        assert!(w.is_hit(0.5)); // >= threshold is a hit
        assert!(!w.is_hit(0.3));
        assert!(!w.is_hit(0.0));

        let strict = test_worker(true, "deberta", 0.9);
        assert!(!strict.is_hit(0.5));
        assert!(strict.is_hit(0.95));
    }

    #[test]
    fn disabled_worker_never_reviews() {
        // A disabled worker is a no-op: review() returns None without spawning.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let w = test_worker(false, "deberta", 0.5);
        let r = rt.block_on(w.review("anything"));
        assert!(r.is_none());
    }

    #[test]
    fn review_round_trip_with_mock_worker() {
        // End-to-end protocol check against the real worker in "mock" mode
        // (no models needed): request id is echoed, risk comes back, and a
        // healthy worker stays alive across calls (no desync / no kill).
        let rt = tokio::runtime::Runtime::new().unwrap();
        let w = test_worker(true, "mock", 0.5);
        let r = rt.block_on(w.review("hello world"));
        assert_eq!(r, Some(0.9));
        let r2 = rt.block_on(w.review("second message"));
        assert_eq!(r2, Some(0.9));
    }

    #[test]
    fn compose_rejected_carries_reason_raw_processed() {
        let chat = ChatMessage {
            platform: "twitch".into(),
            raw_data: vec![],
            raw_message: "b4dx".into(),
            user_uuid7: "u1".into(),
            command: None,
            user_data: None,
            channel_id: "chan".into(),
        };
        let rej = compose_rejected("uuid-9", &chat, "b*d*", "banned word 'x' via leet");
        assert_eq!(rej.message_uuid7, "uuid-9");
        assert_eq!(rej.origin, "banned-words");
        assert_eq!(rej.reason, "banned word 'x' via leet");
        assert_eq!(rej.processed_message, Some("b*d*".to_string()));
        let raw = rej.message.unwrap();
        assert_eq!(raw.raw_message, "b4dx");
        assert_eq!(raw.platform, "twitch");
    }

    #[test]
    fn apply_censor_sentence_and_token_modes() {
        let mut cfg = cfg_with_review();
        cfg.sentence_mode = "censor".to_string();
        assert_eq!(apply_censor(&cfg, "some badword here"), "*".repeat(17));

        cfg.sentence_mode = "replace".to_string();
        cfg.replace_sentence = "[removed]".to_string();
        assert_eq!(apply_censor(&cfg, "some badword here"), "[removed]");

        cfg.sentence_mode = "none".to_string();
        cfg.replace_sentence = String::new();
        cfg.censor_mode = "hard".to_string();
        assert_eq!(apply_censor(&cfg, "badword is a word"), "******* is a word");
    }
}

/// Table-driven coverage of banned_words' censorship: detection variants, the
/// per-token censor modes (soft/hard/mid/replace), the whole-message "sentence"
/// modes (replace message / censor message), and the replace-word /
/// replace-sentence paths. ADD A ROW to a table to add a case — that is the
/// point: dozens of cases, one line each.
#[cfg(test)]
mod censor_matrix {
    use super::*;

    fn list(words: &[&str]) -> Vec<String> {
        words.iter().map(|s| s.to_string()).collect()
    }

    /// A config with only the censor-relevant fields set; everything else is the
    /// module default.
    fn cfg(
        banned: &[&str],
        censor_mode: &str,
        replace_word: &str,
        sentence_mode: &str,
        replace_sentence: &str,
    ) -> Config {
        Config {
            banned_words: list(banned),
            censor_mode: censor_mode.to_string(),
            replace_word: replace_word.to_string(),
            sentence_mode: sentence_mode.to_string(),
            replace_sentence: replace_sentence.to_string(),
            ..default_config()
        }
    }

    /// Detection variants: token exact / leet / no-spaces, and the whole-message
    /// split fallback.
    #[test]
    fn detection_variants() {
        // (label, message, banned, expected (word, variant) | None)
        let cases: Vec<(&str, &str, Vec<&str>, Option<(&str, &str)>)> = vec![
            ("clean", "hello friends", vec!["badword"], None),
            ("exact", "you are a badword", vec!["badword"], Some(("badword", "exact"))),
            ("upper", "BADWORD", vec!["badword"], Some(("badword", "exact"))),
            ("mixed-case", "BaDwOrD", vec!["badword"], Some(("badword", "exact"))),
            ("leet-digits", "b4dw0rd", vec!["badword"], Some(("badword", "normalized"))),
            ("leet-symbols", "b@dw0rd", vec!["badword"], Some(("badword", "normalized"))),
            ("split-spaces", "b a d w o r d", vec!["badword"], Some(("badword", "normalized"))),
            ("split-underscores", "b_a_d_w_o_r_d", vec!["badword"], Some(("badword", "normalized"))),
            ("substring-class", "class", vec!["ass"], None),
            ("substring-grass", "grass", vec!["ass"], None),
            ("substring-hello", "hello", vec!["hell"], None),
            ("standalone-short-word", "you ass", vec!["ass"], Some(("ass", "exact"))),
            ("leet-hell", "go to h3ll", vec!["hell"], Some(("hell", "normalized"))),
            ("empty-message", "", vec!["badword"], None),
            ("empty-word-is-skipped", "anything at all", vec![""], None),
            ("first-of-many-wins", "shoot the badword", vec!["shoot", "badword"], Some(("shoot", "exact"))),
        ];
        for (label, msg, banned, expect) in cases {
            let got = detect_banned(msg, &list(&banned));
            match expect {
                None => assert!(got.is_none(), "[{label}] expected nothing, got {got:?}"),
                Some((w, v)) => {
                    let (gw, gv) =
                        got.unwrap_or_else(|| panic!("[{label}] expected {w}/{v}, got None"));
                    assert_eq!((gw.as_str(), gv), (w, v), "[{label}]");
                }
            }
        }
    }

    /// Per-token censor modes ("censor word"): soft / hard / mid / replace, plus
    /// the length edges and the unknown-mode fallback.
    #[test]
    fn per_token_modes() {
        // (label, token, mode, replace_word, expected)
        let cases: Vec<(&str, &str, &str, &str, &str)> = vec![
            ("soft-7", "badword", "soft", "", "b*****d"),
            ("soft-3", "abc", "soft", "", "a*c"),
            ("soft-2", "ab", "soft", "", "**"),
            ("soft-1", "a", "soft", "", "*"),
            ("soft-keeps-case", "BaDwOrD", "soft", "", "B*****D"),
            ("hard-7", "badword", "hard", "", "*******"),
            ("hard-1", "a", "hard", "", "*"),
            ("mid-7", "badword", "mid", "", "b******"),
            ("mid-1", "a", "mid", "", "a"),
            ("replace-with-word", "badword", "replace", "heck", "heck"),
            ("replace-empty-stars", "badword", "replace", "", "*******"),
            ("unknown-mode-falls-back-to-soft", "badword", "bogus", "", "b*****d"),
        ];
        for (label, tok, mode, rep, want) in cases {
            assert_eq!(censor_token(tok, mode, rep), *want, "[{label}]");
        }
    }

    /// Token censoring inside a message: only matching tokens change; whitespace
    /// and non-matching text survive exactly.
    #[test]
    fn token_censoring_in_messages() {
        // (label, message, banned, mode, replace_word, expected)
        let cases: Vec<(&str, &str, Vec<&str>, &str, &str, &str)> = vec![
            ("only-matching-token", "badword is a word", vec!["badword"], "hard", "", "******* is a word"),
            ("no-false-substring", "grass is ass", vec!["ass"], "hard", "", "grass is ***"),
            ("double-space-preserved", "badword  badword", vec!["badword"], "soft", "", "b*****d  b*****d"),
            ("leading-space", "  badword", vec!["badword"], "hard", "", "  *******"),
            ("trailing-space", "badword  ", vec!["badword"], "hard", "", "*******  "),
            ("leet-token", "b4dw0rd ok", vec!["badword"], "hard", "", "******* ok"),
            ("clean-untouched", "clean text here", vec!["badword"], "hard", "", "clean text here"),
            ("multiple-banned", "badword and shoot", vec!["badword", "shoot"], "hard", "", "******* and *****"),
            ("multibyte-space-delimiter", "badword\u{3000}ok", vec!["badword"], "hard", "", "*******\u{3000}ok"),
        ];
        for (label, msg, banned, mode, rep, want) in cases {
            assert_eq!(censor_message(msg, &list(&banned), mode, rep), *want, "[{label}]");
        }
    }

    /// "Replace word": every matching token becomes the same replacement; note a
    /// repeat is replaced each time (not collapsed).
    #[test]
    fn replace_word_cases() {
        // (label, message, banned, replace_word, expected)
        let cases: Vec<(&str, &str, Vec<&str>, &str, &str)> = vec![
            ("single", "badword", vec!["badword"], "heck", "heck"),
            ("repeat-each-time", "badword and badword", vec!["badword"], "heck", "heck and heck"),
            ("two-different-banned", "badword and shoot", vec!["badword", "shoot"], "[redacted]", "[redacted] and [redacted]"),
            ("non-matching-preserved", "clean badword text", vec!["badword"], "X", "clean X text"),
            ("empty-replacement-stars", "badword", vec!["badword"], "", "*******"),
        ];
        for (label, msg, banned, rep, want) in cases {
            let c = cfg(&banned, "replace", rep, "none", "");
            assert_eq!(apply_censor(&c, msg), *want, "[{label}]");
        }
    }

    /// "Replace message" / whole-sentence modes, and the token fallback.
    #[test]
    fn sentence_mode_cases() {
        let msg = "a badword here";
        let cases: Vec<(&str, Config, &str, String)> = vec![
            ("token-default", cfg(&["badword"], "hard", "", "none", ""), msg, "a ******* here".to_string()),
            ("sentence-censor", cfg(&["badword"], "hard", "", "censor", ""), msg, "*".repeat(msg.chars().count())),
            ("sentence-replace", cfg(&["badword"], "hard", "", "replace", "[removed]"), msg, "[removed]".to_string()),
            ("sentence-replace-fallback-to-token", cfg(&["badword"], "soft", "", "replace", ""), msg, "a b*****d here".to_string()),
            ("sentence-censor-counts-chars-not-bytes", cfg(&["badword"], "hard", "", "censor", ""), "bäd word", "*".repeat("bäd word".chars().count())),
        ];
        for (label, c, input, want) in cases {
            assert_eq!(apply_censor(&c, input), want, "[{label}]");
        }
    }

    /// effective_text: an earlier module's edit wins; only empty/whitespace
    /// incoming falls back to the raw message.
    #[test]
    fn effective_text_cases() {
        // (label, raw, incoming, expected)
        let cases: Vec<(&str, &str, &str, &str)> = vec![
            ("incoming-wins", "raw badword", "cleaned text", "cleaned text"),
            ("empty-incoming-falls-back", "raw badword", "", "raw badword"),
            ("whitespace-incoming-falls-back", "raw badword", "   ", "raw badword"),
            ("both-empty", "", "", ""),
        ];
        for (label, raw, incoming, want) in cases {
            assert_eq!(effective_text(raw, incoming), want, "[{label}]");
        }
    }

    /// The evasions the CONDENSED matcher now catches: attached punctuation,
    /// leet symbols, accents, and non-space separators — the bypasses the old
    /// token-exact matcher missed.
    #[test]
    fn evasions_are_caught() {
        let bad = list(&["badword"]);

        for (label, msg) in [
            ("trailing-bang", "badword!"),
            ("leading-bang", "!badword"),
            ("punctuation-inside", "bad-word"),
            ("dots-between", "b.a.d.w.o.r.d"),
            ("spaces-between", "b a d w o r d"),
            ("underscores-between", "b_a_d_w_o_r_d"),
            ("uppercase", "BADWORD"),
            ("leet-digits", "b4dw0rd"),
            ("leet-at", "b@dw0rd"),
            ("accented", "bädword"),
            ("quoted", "\"badword\""),
        ] {
            assert!(
                detect_banned(msg, &bad).is_some(),
                "[{label}] {msg:?} should be caught"
            );
        }

        // Censoring matches too: the offending span becomes stars, the
        // surrounding punctuation survives untouched.
        assert_eq!(censor_message("badword!", &bad, "hard", ""), "*******!");
        assert_eq!(censor_message("!badword", &bad, "hard", ""), "!*******");
        assert_eq!(censor_message("\"badword\"", &bad, "hard", ""), "\"*******\"");
        assert_eq!(censor_message("b.a.d.w.o.r.d", &bad, "hard", ""), "*************");
        assert_eq!(censor_message("BÄDWORD", &bad, "hard", ""), "*******");

        // Guard: a real substring inside a longer word is still NOT caught.
        for word in ["class", "grass", "hello", "a classic grassy hello"] {
            assert!(
                detect_banned(word, &list(&["ass", "hell"])).is_none(),
                "{word:?} must not be flagged"
            );
        }
    }
}

/// The lightweight AI risk model: a table of (label, text, high?) cases. The
/// score is 0..1; "high" means it should clear a 0.5 threshold.
#[cfg(test)]
mod risk_model_tests {
    use super::*;

    #[test]
    fn risk_score_is_in_range_and_sensible() {
        // (label, text, expect_high)
        let cases: Vec<(&str, &str, bool)> = vec![
            ("empty", "", false),
            ("whitespace", "   ", false),
            ("greeting", "hello everyone, how are you?", false),
            ("small talk", "good game well played", false),
            ("thanks", "thanks for the stream, love it", false),
            ("punct only", "!!???...", false),
            ("toxic-insult", "you are a stupid idiot", true),
            ("toxic-shutup", "shut up moron", true),
            ("toxic-loser", "what a loser", true),
            ("toxic-kys", "kys", true),
            ("obf-spaced", "s t u p i d", true),
            ("obf-caps", "STUPID!!!", true),
            ("obf-punct", "idiot.", true),
            ("obf-leet", "m0r0n", true),
            ("obf-dots", "i.d.i.o.t", true),
        ];
        for (label, text, high) in cases {
            let r = risk_score(text);
            assert!(
                (0.0..=1.0).contains(&r),
                "[{label}] {text:?} scored {r}, out of 0..1"
            );
            if high {
                assert!(r >= 0.5, "[{label}] {text:?} scored {r}, expected high (>=0.5)");
            } else {
                assert!(r < 0.3, "[{label}] {text:?} scored {r}, expected low (<0.3)");
            }
        }
    }

    /// More seed hits → higher risk (monotonic in the lexicon signal).
    #[test]
    fn more_toxicity_scores_higher() {
        let one = risk_score("idiot");
        let two = risk_score("idiot moron");
        assert!(two > one, "two insults ({two}) should outscore one ({one})");
        assert!(risk_score("you stupid fucking idiot") >= two);
    }

    /// The model is a pure function: same input, same output, no state.
    #[test]
    fn deterministic() {
        for t in ["hello", "you idiot", "s t u p i d", ""] {
            assert_eq!(risk_score(t), risk_score(t));
        }
    }
}
