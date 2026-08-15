//! Managed runtime assets shared by every front end.

use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// Stable filename used by the RVC ContentVec/HuBERT checkpoint.
pub const HUBERT_FILENAME: &str = "hubert_base.pt";

/// Immutable upstream revision containing the legacy RVC ContentVec checkpoint.
const HUBERT_URL: &str = concat!(
    "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/",
    "1c75048c96f23f99da4b12909b532b5983290d7d/hubert_base.pt"
);
const HUBERT_SIZE: u64 = 189_507_909;
const HUBERT_SHA256: &str = "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96";
static TEMPORARY_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Failure while locating, downloading, or validating a managed model asset.
#[derive(Debug, Error)]
pub enum AssetError {
    /// The platform did not expose a durable per-user cache directory.
    #[error("cannot determine the per-user cache directory for managed models")]
    CacheDirectoryUnavailable,
    /// Local cache I/O failed.
    #[error("managed model cache I/O failed: {0}")]
    Io(#[from] std::io::Error),
    /// The HTTPS transfer failed.
    #[error("failed to download the managed HuBERT model: {0}")]
    Download(String),
    /// The received or cached asset has an unexpected byte length.
    #[error("managed HuBERT model has {actual} bytes; expected {expected}")]
    WrongSize {
        /// Observed byte length.
        actual: u64,
        /// Required byte length.
        expected: u64,
    },
    /// The received or cached asset does not match the pinned model.
    #[error("managed HuBERT model checksum mismatch; expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Observed SHA-256 digest.
        actual: String,
        /// Required SHA-256 digest.
        expected: &'static str,
    },
}

/// Returns the canonical per-user location of the managed HuBERT model.
pub fn hubert_cache_path() -> Result<PathBuf, AssetError> {
    Ok(platform_cache_root()?
        .join("rvc-rs")
        .join("models")
        .join(HUBERT_FILENAME))
}

/// Reports whether a complete, authentic managed HuBERT model is already cached.
pub fn hubert_is_ready() -> bool {
    hubert_cache_path()
        .and_then(|path| verify_hubert(&path))
        .is_ok()
}

/// Reports whether a managed HuBERT file exists without hashing its full contents.
///
/// This is intended only for progress messages. [`ensure_hubert`] always performs
/// the authoritative size and SHA-256 validation before inference.
pub fn hubert_is_cached() -> bool {
    hubert_cache_path().is_ok_and(|path| path.is_file())
}

/// Resolves the mandatory HuBERT model, downloading it once when absent or invalid.
///
/// Every inference path calls this function. Front ends never accept a user-selected
/// ContentVec path, so all voices use the exact checkpoint pinned by this crate.
pub fn ensure_hubert() -> Result<PathBuf, AssetError> {
    let destination = hubert_cache_path()?;
    if verify_hubert(&destination).is_ok() {
        return Ok(destination);
    }

    let parent = destination
        .parent()
        .ok_or(AssetError::CacheDirectoryUnavailable)?;
    fs::create_dir_all(parent)?;
    let sequence = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temporary = parent.join(format!(
        "{HUBERT_FILENAME}.part-{}-{sequence}",
        std::process::id()
    ));

    let result = download_hubert(&temporary);
    if let Err(error) = result {
        let _ = fs::remove_file(&temporary);
        return Err(error);
    }

    // Another process may have completed the same managed download while this
    // process was transferring. Prefer its already-verified destination.
    if verify_hubert(&destination).is_ok() {
        let _ = fs::remove_file(&temporary);
        return Ok(destination);
    }
    if destination.exists() {
        fs::remove_file(&destination)?;
    }
    fs::rename(&temporary, &destination)?;
    verify_hubert(&destination)?;
    Ok(destination)
}

fn download_hubert(destination: &Path) -> Result<(), AssetError> {
    let response = ureq::get(HUBERT_URL)
        .call()
        .map_err(|error| AssetError::Download(error.to_string()))?;
    let mut reader = response.into_body().into_reader();
    let mut file = File::create(destination)?;
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        total = total.saturating_add(read as u64);
        if total > HUBERT_SIZE {
            return Err(AssetError::WrongSize {
                actual: total,
                expected: HUBERT_SIZE,
            });
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])?;
    }
    file.flush()?;
    file.sync_all()?;
    verify_digest(total, &format!("{:x}", hasher.finalize()))
}

fn verify_hubert(path: &Path) -> Result<(), AssetError> {
    let mut file = File::open(path)?;
    let length = file.metadata()?.len();
    if length != HUBERT_SIZE {
        return Err(AssetError::WrongSize {
            actual: length,
            expected: HUBERT_SIZE,
        });
    }

    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    verify_digest(length, &format!("{:x}", hasher.finalize()))
}

fn verify_digest(length: u64, digest: &str) -> Result<(), AssetError> {
    if length != HUBERT_SIZE {
        return Err(AssetError::WrongSize {
            actual: length,
            expected: HUBERT_SIZE,
        });
    }
    if digest != HUBERT_SHA256 {
        return Err(AssetError::ChecksumMismatch {
            actual: digest.to_owned(),
            expected: HUBERT_SHA256,
        });
    }
    Ok(())
}

#[cfg(target_os = "windows")]
fn platform_cache_root() -> Result<PathBuf, AssetError> {
    env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .ok_or(AssetError::CacheDirectoryUnavailable)
}

#[cfg(target_os = "macos")]
fn platform_cache_root() -> Result<PathBuf, AssetError> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join("Library").join("Caches"))
        .ok_or(AssetError::CacheDirectoryUnavailable)
}

#[cfg(all(unix, not(target_os = "macos")))]
fn platform_cache_root() -> Result<PathBuf, AssetError> {
    if let Some(path) = env::var_os("XDG_CACHE_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path));
    }
    env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".cache"))
        .ok_or(AssetError::CacheDirectoryUnavailable)
}

#[cfg(not(any(unix, target_os = "windows")))]
fn platform_cache_root() -> Result<PathBuf, AssetError> {
    Err(AssetError::CacheDirectoryUnavailable)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_asset_metadata_is_self_consistent() {
        assert_eq!(HUBERT_SHA256.len(), 64);
        assert!(HUBERT_SIZE > 100_000_000);
        assert!(HUBERT_URL.contains("1c75048c96f23f99da4b12909b532b5983290d7d"));
    }

    #[test]
    fn rejects_an_unexpected_digest() {
        assert!(matches!(
            verify_digest(HUBERT_SIZE, &"0".repeat(64)),
            Err(AssetError::ChecksumMismatch { .. })
        ));
    }
}
