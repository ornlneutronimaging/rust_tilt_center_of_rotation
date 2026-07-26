//! Append-only log in the shared VENUS log directory, following the
//! `<tool>_<user>.log` convention of the other imaging tools. Best-effort:
//! if the file cannot be opened, calls are silently dropped.

use chrono::Local;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::{Mutex, OnceLock};

const LOG_DIR: &str = "/SNS/VENUS/shared/log";

static SINK: OnceLock<Option<Mutex<File>>> = OnceLock::new();

pub fn user_id() -> String {
    std::env::var("USER").unwrap_or_else(|_| "user".to_owned())
}

/// `/SNS/VENUS/shared/log/rust_tilt_center_of_rotation_<user>.log`
pub fn log_path() -> PathBuf {
    PathBuf::from(LOG_DIR).join(format!("rust_tilt_center_of_rotation_{}.log", user_id()))
}

pub fn init() {
    SINK.get_or_init(|| {
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(log_path())
            .ok()
            .map(Mutex::new)
    });
    log(format!(
        "=== Application started (user: {}, pid: {}, version: {}) ===",
        user_id(),
        std::process::id(),
        env!("CARGO_PKG_VERSION"),
    ));
}

fn write_line(level: &str, msg: &str) {
    if let Some(mutex) = SINK.get().and_then(|s| s.as_ref())
        && let Ok(mut file) = mutex.lock()
    {
        let ts = Local::now().format("%Y-%m-%d %H:%M:%S,%3f");
        let _ = writeln!(file, "{ts} - {level} - {msg}");
        let _ = file.flush();
    }
}

pub fn log(msg: impl AsRef<str>) {
    write_line("INFO", msg.as_ref());
}

pub fn error(msg: impl AsRef<str>) {
    write_line("ERROR", msg.as_ref());
}
