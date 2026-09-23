
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

use cockatiel_client::proto::container::Payload;
use cockatiel_client::proto::*;
use cockatiel_client::{CockatielClient, PromptKind};

type WsWriteHalf = futures_util::stream::SplitSink<
    tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
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
    llm_review_threshold: f64,
    /// "deberta" (default) | "llama-guard" | "auto" — both optional.
    #[serde(default = "default_review_engine")]
    llm_review_engine: String,
}

fn default_mode() -> String {
    "soft".to_string()
}

fn default_sentence_mode() -> String {
    "none".to_string()
}

fn default_review_threshold() -> f64 {
    0.5
}

fn default_review_engine() -> String {
    "deberta".to_string()
}

/// Leet-speak translation table (common substitutions).
fn leet_translate(s: &str) -> String {
    s.chars()
        .map(|c| match c.to_ascii_lowercase() {
            '3' => 'e',
            '4' | '@' => 'a',
            '0' => 'o',
            '1' | '!' => 'i',
            '5' | '$' => 's',
            '7' => 't',
            'z' => 's',
            other => other,
        })
        .collect()
}

/// Collapse whitespace and remove all spaces (for "no-spaces" / "extra-spaces" detection).
fn strip_all_spaces(s: &str) -> String {
    s.chars().filter(|c| !c.is_whitespace() && *c != '_').collect()
}

/// Check if the message contains any banned word, across variants.
/// Returns the first banned word matched and which variant it was.
fn detect_banned(message: &str, banned: &[String]) -> Option<(String, &'static str)> {
    let lowered = message.to_lowercase();
    for word in banned {
        let w = word.to_lowercase();
        if w.is_empty() {
            continue;
        }
        // Token-level detection: for each whitespace-delimited token, compare
        // the token (original, leet-translated, and space-stripped) to the word.
        for token in lowered.split_whitespace() {
            if token == w {
                return Some((w.clone(), "exact"));
            }
            let leet = leet_translate(token);
            if leet == w {
                return Some((w.clone(), "leet"));
            }
            let nospace = strip_all_spaces(token);
            if nospace == w || nospace.contains(&w) {
                return Some((w.clone(), "no-spaces"));
            }
        }
        // Whole-message fallback: a space-stripped message containing the word.
        let stripped = strip_all_spaces(&lowered);
        if stripped.contains(&w) {
            return Some((w.clone(), "no-spaces"));
        }
    }
    None
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
    let mut result = String::new();
    let mut rest = message;
    while !rest.is_empty() {
        // Find the next whitespace-delimited token.
        let mut end = rest.len();
        for (i, ch) in rest.char_indices() {
            if ch.is_whitespace() {
                end = i;
                break;
            }
        }
        let (token, remainder) = rest.split_at(end);
        let matched = banned.iter().any(|word| {
            let w = word.to_lowercase();
            if w.is_empty() {
                return false;
            }
            let t = token.to_lowercase();
            t == w
                || leet_translate(&t) == w
                || strip_all_spaces(&t) == w
                || strip_all_spaces(&t).contains(&w)
        });
        if matched {
            result.push_str(&censor_token(token, mode, replace_word));
        } else {
            result.push_str(token);
        }
        result.push_str(&remainder[..0]); // empty
        rest = &remainder;
        // Preserve the delimiter (the split_at consumed it as part of remainder).
        if !rest.is_empty() && end < message.len() {
            // remainder starts with the whitespace that ended the token.
            result.push_str(&rest[..1]);
            rest = &rest[1..];
        }
    }
    result
}

/// Optional LLM review backend. Spawns `worker/review_worker.py` lazily (only
/// when enabled), feeds it JSONL requests on stdin, reads the 0–1 risk back.
/// A review that times out (slow model/load) yields None and the message is
/// passed through unreviewed — the module never blocks the pipeline ack.
struct ReviewWorker {
    enabled: bool,
    engine: String,
    threshold: f64,
    child: AsyncMutex<Option<ReviewChild>>,
    dead: Arc<AtomicBool>,
}

struct ReviewChild {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl ReviewWorker {
    fn new(enabled: bool, engine: String, threshold: f64) -> Self {
        Self {
            enabled,
            engine,
            threshold,
            child: AsyncMutex::new(None),
            dead: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Score a message. None = disabled, worker failed to start, timed out, or
    /// an error — the caller passes the message through unreviewed.
    async fn review(&self, text: &str) -> Option<f64> {
        if !self.enabled || self.dead.load(Ordering::SeqCst) {
            return None;
        }
        let mut guard = self.child.lock().await;
        if guard.is_none() {
            match spawn_review_worker(&self.engine).await {
                Ok(c) => *guard = Some(c),
                Err(e) => {
                    warn!("llm_review: worker failed to start ({}); disabling review", e);
                    self.dead.store(true, Ordering::SeqCst);
                    return None;
                }
            }
        }
        let child = guard.as_mut()?;
        let req = serde_json::json!({ "text": text }).to_string();
        if child.stdin.write_all(req.as_bytes()).await.is_err() || child.stdin.write_all(b"\n").await.is_err() {
            self.dead.store(true, Ordering::SeqCst);
            return None;
        }
        let mut line = String::new();
        match tokio::time::timeout(Duration::from_secs(2), child.stdout.read_line(&mut line)).await {
            Ok(Ok(n)) if n > 0 => {
                match serde_json::from_str::<serde_json::Value>(&line) {
                    Ok(v) => {
                        if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                            // The worker is broken (model missing / HF-gated): warn
                            // once + disable so we stop paying per-message timeouts.
                            warn!("llm_review worker error ({}); disabling review", err);
                            self.dead.store(true, Ordering::SeqCst);
                            None
                        } else {
                            v.get("risk").and_then(|r| r.as_f64())
                        }
                    }
                    Err(_) => None,
                }
            }
            _ => None, // timeout or worker gone
        }
    }

    /// Whether this text should be treated as a hit given the configurable
    /// risk-certainty threshold.
    fn is_hit(&self, risk: f64) -> bool {
        risk >= self.threshold
    }
}

async fn spawn_review_worker(engine: &str) -> Result<ReviewChild, Box<dyn std::error::Error>> {
    let mut cmd = Command::new("python3");
    cmd.arg("worker/review_worker.py")
        .arg("--engine")
        .arg(engine)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::inherit());
    let mut child = cmd.spawn()?;
    let stdin = child.stdin.take().ok_or("no stdin")?;
    let stdout = child.stdout.take().ok_or("no stdout")?;
    Ok(ReviewChild { stdin, stdout: BufReader::new(stdout) })
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
        processed_message: processed.to_string(),
        reason: reason.to_string(),
        origin: "banned-words".to_string(),
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
                censor_message(original, &config.banned_words, &config.censor_mode, &config.replace_word)
            } else {
                config.replace_sentence.clone()
            }
        }
        // Default: censor only the offending token(s).
        _ => censor_message(original, &config.banned_words, &config.censor_mode, &config.replace_word),
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
    let container = Container {
        version: 1,
        auth_token: auth_token.to_string(),
        module_name: module_name.to_string(),
        module_instance_uuid7: instance_uuid.to_string(),
        payload: Some(Payload::Prompt(prompt)),
    };
    let mut buf = Vec::new();
    if container.encode(&mut buf).is_err() {
        return None;
    }
    let mut guard = write_ws.lock().await;
    if guard.send(WsMessage::Binary(buf.into())).await.is_err() {
        return None;
    }
    drop(guard);

    let deadline = tokio::time::Instant::now() + Duration::from_secs(timeout as u64 + 10);
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
        60,
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
        60,
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

    // Connect to the engine as a preprocess module (CLI overrides: --ip/--port/--pin).
    let client = CockatielClient::connect("banned_words.json").await?;
    let (write, mut read) = client.stream.split();
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
    // message pre-processing. Spawned BEFORE load_config so prompts work.
    {
        let prompt_tx_task = prompt_tx.clone();
        let config_shared = Arc::clone(&config_shared);
        let review_worker_shared = Arc::clone(&review_worker_shared);
        let write_shared = Arc::clone(&write_shared);
        let auth_token = auth_token.clone();
        let module_name = module_name.clone();
        let instance_uuid = instance_uuid.clone();
        tokio::spawn(async move {
            while let Some(msg) = read.next().await {
                let Ok(WsMessage::Binary(data)) = msg else { continue };
                let Ok(container) = Container::decode(data.as_ref()) else { continue };

                match container.payload {
                    Some(Payload::AuthVerify(_)) => {
                        // Answer the engine's liveness probe (this module reads
                        // the socket directly, so the client's auto-answer is
                        // bypassed — without this the watchdog severs us).
                        let reply = Container {
                            version: 1,
                            auth_token: auth_token.clone(),
                            module_name: module_name.clone(),
                            module_instance_uuid7: instance_uuid.clone(),
                            payload: Some(Payload::AuthVerify(AuthVerify {
                                cur_auth: auth_token.clone(),
                            })),
                        };
                        let mut buf = Vec::new();
                        if reply.encode(&mut buf).is_ok() {
                            let mut w = write_shared.lock().await;
                            let _ = w.send(WsMessage::Binary(buf.into())).await;
                        }
                    }
                    Some(Payload::PromptResponse(resp)) => {
                        // Forward operator answers to the awaiting prompt.
                        let _ = prompt_tx_task.send(resp);
                    }
                    Some(Payload::MessagePreProcess(pre)) => {
                        let Some(chat) = &pre.raw_message else { continue };
                        let config = config_shared.lock().unwrap().clone();
                        let original = chat.raw_message.clone();
                        let uuid = pre.message_uuid7.clone();
                        let worker = review_worker_shared.lock().unwrap().clone();

                        let mut flagged = false;
                        let mut reason = String::new();
                        match detect_banned(&original, &config.banned_words) {
                            Some((word, variant)) => {
                                warn!("Banned word '{}' detected via {} variant", word, variant);
                                flagged = true;
                                reason = format!("banned word '{}' via {} variant", word, variant);
                            }
                            None if config.llm_review => {
                                // Optional LLM review: catch what the word list
                                // missed. Risk >= the configurable certainty
                                // threshold is treated exactly like a word hit.
                                if let Some(w) = &worker {
                                    if let Some(risk) = w.review(&original).await {
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
                            _ => {}
                        }

                        let censored = if flagged {
                            apply_censor(&config, &original)
                        } else {
                            original.clone()
                        };

                        // If flag_for_review, send a ChatMessageRejected so the
                        // engine logs the rejection clearly (reason + raw +
                        // processed) instead of an obscure log string.
                        if flagged && config.flag_for_review {
                            let rej = compose_rejected(&uuid, chat, &censored, &reason);
                            let log = Container {
                                version: 1,
                                auth_token: auth_token.clone(),
                                module_name: module_name.clone(),
                                module_instance_uuid7: instance_uuid.clone(),
                                payload: Some(Payload::ChatMessageRejected(rej)),
                            };
                            let mut buf = Vec::new();
                            if log.encode(&mut buf).is_ok() {
                                let mut w = write_shared.lock().await;
                                let _ = w.send(WsMessage::Binary(buf.into())).await;
                            }
                        }

                        // Reply with the (possibly censored) message, same uuid → engine acks pre_process.
                        let reply = Container {
                            version: 1,
                            auth_token: auth_token.clone(),
                            module_name: module_name.clone(),
                            module_instance_uuid7: instance_uuid.clone(),
                            payload: Some(Payload::MessagePreProcess(MessagePreProcess {
audio: Vec::new(),
                        audio_type: String::new(),
                                message_uuid7: uuid.clone(),
                                raw_message: Some(ChatMessage {
                                    platform: chat.platform.clone(),
                                    raw_data: chat.raw_data.clone(),
                                    raw_message: censored.clone(),
                                    user_uuid7: chat.user_uuid7.clone(),
                                    command: chat.command.clone(),
                                    channel_id: chat.channel_id.clone(),
                                    user_data: chat.user_data.clone(),
                                }),
                            })),
                        };
                        let mut buf = Vec::new();
                        if reply.encode(&mut buf).is_ok() {
                            let mut w = write_shared.lock().await;
                            let _ = w.send(WsMessage::Binary(buf.into())).await;
                            if flagged {
                                info!("Censored message ({}): {:?} -> {:?}", uuid, original, censored);
                            }
                        }
                    }
                    _ => {}
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

    fn cfg_with_review() -> Config {
        Config {
            llm_review: true,
            llm_review_threshold: 0.5,
            llm_review_engine: "deberta".to_string(),
            ..default_config()
        }
    }

    #[test]
    fn llm_review_off_by_default() {
        assert!(!default_config().llm_review);
        assert_eq!(default_config().llm_review_threshold, 0.5);
        assert_eq!(default_config().llm_review_engine, "deberta");
    }

    #[test]
    fn threshold_hit_and_miss() {
        let w = ReviewWorker::new(true, "deberta".to_string(), 0.5);
        assert!(w.is_hit(0.9));
        assert!(w.is_hit(0.5)); // >= threshold is a hit
        assert!(!w.is_hit(0.3));
        assert!(!w.is_hit(0.0));

        let strict = ReviewWorker::new(true, "deberta".to_string(), 0.9);
        assert!(!strict.is_hit(0.5));
        assert!(strict.is_hit(0.95));
    }

    #[test]
    fn disabled_worker_never_reviews() {
        // A disabled worker is a no-op: review() returns None without spawning.
        let rt = tokio::runtime::Runtime::new().unwrap();
        let w = ReviewWorker::new(false, "deberta".to_string(), 0.5);
        let r = rt.block_on(w.review("anything"));
        assert!(r.is_none());
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
        assert_eq!(rej.processed_message, "b*d*");
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
