//! Managed runtime assets shared by every front end.

use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

/// A pinned model file that the engine downloads and verifies on demand.
#[derive(Clone, Copy, Debug)]
pub struct ManagedAsset {
    /// Human-readable name used in progress and error messages.
    pub label: &'static str,
    /// Cache filename.
    pub filename: &'static str,
    url: &'static str,
    size: u64,
    sha256: &'static str,
}

/// RVC ContentVec/HuBERT content encoder.
pub const HUBERT: ManagedAsset = ManagedAsset {
    label: "HuBERT",
    filename: "hubert_base.pt",
    url: concat!(
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/",
        "1c75048c96f23f99da4b12909b532b5983290d7d/hubert_base.pt"
    ),
    size: 189_507_909,
    sha256: "f54b40fd2802423a5643779c4861af1e9ee9c1564dc9d32f54f20b5ffba7db96",
};

/// RVC RMVPE pitch estimator.
pub const RMVPE: ManagedAsset = ManagedAsset {
    label: "RMVPE",
    filename: "rmvpe.pt",
    url: concat!(
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/",
        "e6d0c1a17da07c33557852f9dfa2bd44cc75737d/rmvpe.pt"
    ),
    size: 181_184_272,
    sha256: "6d62215f4306e3ca278246188607209f09af3dc77ed4232efdd069798c4ec193",
};

const DOWNLOAD_ATTEMPTS: usize = 3;
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
    #[error("failed to download the managed {asset} model: {message}")]
    Download {
        /// Asset label.
        asset: &'static str,
        /// Transport error text.
        message: String,
    },
    /// The received or cached asset has an unexpected byte length.
    #[error("managed {asset} model has {actual} bytes; expected {expected}")]
    WrongSize {
        /// Asset label.
        asset: &'static str,
        /// Observed byte length.
        actual: u64,
        /// Required byte length.
        expected: u64,
    },
    /// The received or cached asset does not match the pinned model.
    #[error("managed {asset} model checksum mismatch; expected {expected}, got {actual}")]
    ChecksumMismatch {
        /// Asset label.
        asset: &'static str,
        /// Observed SHA-256 digest.
        actual: String,
        /// Required SHA-256 digest.
        expected: &'static str,
    },
}

impl ManagedAsset {
    /// Returns the canonical per-user cache location.
    pub fn cache_path(&self) -> Result<PathBuf, AssetError> {
        Ok(platform_cache_root()?
            .join("rvc-rs")
            .join("models")
            .join(self.filename))
    }

    /// Reports whether a complete, authentic copy is already cached.
    pub fn is_ready(&self) -> bool {
        self.cache_path()
            .and_then(|path| self.verify(&path))
            .is_ok()
    }

    /// Reports whether a cached file exists without hashing it.
    ///
    /// This is intended only for progress messages. [`ManagedAsset::ensure`]
    /// always performs the authoritative size and SHA-256 validation.
    pub fn is_cached(&self) -> bool {
        self.cache_path().is_ok_and(|path| path.is_file())
    }

    /// Resolves the asset, downloading it once when absent or invalid.
    ///
    /// Front ends never accept user-selected paths for managed assets, so every
    /// conversion uses the exact files pinned by this crate.
    pub fn ensure(&self) -> Result<PathBuf, AssetError> {
        let destination = self.cache_path()?;
        if self.verify(&destination).is_ok() {
            return Ok(destination);
        }

        let parent = destination
            .parent()
            .ok_or(AssetError::CacheDirectoryUnavailable)?;
        fs::create_dir_all(parent)?;
        let sequence = TEMPORARY_COUNTER.fetch_add(1, Ordering::Relaxed);
        let temporary = parent.join(format!(
            "{}.part-{}-{sequence}",
            self.filename,
            std::process::id()
        ));

        let mut result = Ok(());
        for _ in 0..DOWNLOAD_ATTEMPTS {
            result = self.download(&temporary);
            if result.is_ok() {
                break;
            }
        }
        if let Err(error) = result {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }

        // Another process may have completed the same managed download while this
        // process was transferring. Prefer its already-verified destination.
        if self.verify(&destination).is_ok() {
            let _ = fs::remove_file(&temporary);
            return Ok(destination);
        }
        if destination.exists() {
            fs::remove_file(&destination)?;
        }
        fs::rename(&temporary, &destination)?;
        self.verify(&destination)?;
        Ok(destination)
    }

    fn download(&self, destination: &Path) -> Result<(), AssetError> {
        let transfer_error = |message: String| AssetError::Download {
            asset: self.label,
            message,
        };
        let response = ureq::get(self.url)
            .call()
            .map_err(|error| transfer_error(error.to_string()))?;
        let mut reader = response.into_body().into_reader();
        let mut file = File::create(destination)?;
        let mut hasher = Sha256::new();
        let mut total = 0_u64;
        let mut buffer = [0_u8; 64 * 1024];

        loop {
            let read = reader
                .read(&mut buffer)
                .map_err(|error| transfer_error(error.to_string()))?;
            if read == 0 {
                break;
            }
            total = total.saturating_add(read as u64);
            if total > self.size {
                return Err(AssetError::WrongSize {
                    asset: self.label,
                    actual: total,
                    expected: self.size,
                });
            }
            hasher.update(&buffer[..read]);
            file.write_all(&buffer[..read])?;
        }
        file.flush()?;
        file.sync_all()?;
        self.verify_digest(total, &format!("{:x}", hasher.finalize()))
    }

    fn verify(&self, path: &Path) -> Result<(), AssetError> {
        let mut file = File::open(path)?;
        let length = file.metadata()?.len();
        if length != self.size {
            return Err(AssetError::WrongSize {
                asset: self.label,
                actual: length,
                expected: self.size,
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
        self.verify_digest(length, &format!("{:x}", hasher.finalize()))
    }

    fn verify_digest(&self, length: u64, digest: &str) -> Result<(), AssetError> {
        if length != self.size {
            return Err(AssetError::WrongSize {
                asset: self.label,
                actual: length,
                expected: self.size,
            });
        }
        if digest != self.sha256 {
            return Err(AssetError::ChecksumMismatch {
                asset: self.label,
                actual: digest.to_owned(),
                expected: self.sha256,
            });
        }
        Ok(())
    }
}

/// Returns the canonical per-user location of the managed HuBERT model.
pub fn hubert_cache_path() -> Result<PathBuf, AssetError> {
    HUBERT.cache_path()
}

/// Reports whether a complete, authentic managed HuBERT model is already cached.
pub fn hubert_is_ready() -> bool {
    HUBERT.is_ready()
}

/// Reports whether a managed HuBERT file exists without hashing its full contents.
pub fn hubert_is_cached() -> bool {
    HUBERT.is_cached()
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
        for asset in [HUBERT, RMVPE] {
            assert_eq!(asset.sha256.len(), 64);
            assert!(asset.size > 100_000_000);
            assert!(asset.url.ends_with(asset.filename));
        }
        assert!(HUBERT
            .url
            .contains("1c75048c96f23f99da4b12909b532b5983290d7d"));
        assert!(RMVPE
            .url
            .contains("e6d0c1a17da07c33557852f9dfa2bd44cc75737d"));
    }

    #[test]
    fn rejects_an_unexpected_digest() {
        assert!(matches!(
            HUBERT.verify_digest(HUBERT.size, &"0".repeat(64)),
            Err(AssetError::ChecksumMismatch { .. })
        ));
    }
}
