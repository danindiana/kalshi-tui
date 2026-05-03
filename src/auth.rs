use anyhow::{bail, Context, Result};
use base64::{engine::general_purpose::STANDARD as B64, Engine};
use openssl::{hash::MessageDigest, pkey::PKey, rsa::Rsa, sign::Signer};
use std::{collections::HashMap, path::{Path, PathBuf}, time::{SystemTime, UNIX_EPOCH}};

fn config_dir() -> PathBuf {
    if let Ok(s) = std::env::var("KALSHI_TUI_CONFIG_DIR") {
        return PathBuf::from(s);
    }
    if let Ok(s) = std::env::var("XDG_CONFIG_HOME") {
        return PathBuf::from(s).join("kalshi-tui");
    }
    match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h).join(".config").join("kalshi-tui"),
        Err(_) => PathBuf::from(".config/kalshi-tui"),
    }
}

fn config_path() -> PathBuf { config_dir().join("config.toml") }
fn default_key_path() -> PathBuf { config_dir().join("private_key.pem") }

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct KalshiConfig {
    pub key_id: String,
    pub private_key_path: String,
}

impl KalshiConfig {
    pub fn load() -> Result<Self> {
        let cfg_path = config_path();
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Cannot read {}", cfg_path.display()))?;

        let mut map: HashMap<String, String> = HashMap::new();
        for line in raw.lines() {
            let line = line.trim();
            if line.starts_with('#') || line.is_empty() { continue; }
            if let Some((k, v)) = line.split_once('=') {
                map.insert(
                    k.trim().to_string(),
                    v.trim().trim_matches('"').to_string(),
                );
            }
        }

        let key_id = map.get("key_id").cloned().unwrap_or_default();
        if key_id.is_empty() {
            bail!(
                "key_id is empty in {}.\n\
                 Find your key ID in Kalshi portal → Settings → API Keys.",
                cfg_path.display()
            );
        }

        let private_key_path = map
            .get("private_key_path")
            .cloned()
            .unwrap_or_else(|| default_key_path().display().to_string());

        if !Path::new(&private_key_path).exists() {
            bail!("Private key not found at {private_key_path}");
        }

        Ok(KalshiConfig { key_id, private_key_path })
    }

    pub fn is_configured() -> bool {
        Self::load().is_ok()
    }
}

// ── Signer ────────────────────────────────────────────────────────────────────

pub struct KalshiSigner {
    pub key_id: String,
    pkey: PKey<openssl::pkey::Private>,
}

impl KalshiSigner {
    pub fn new(cfg: &KalshiConfig) -> Result<Self> {
        let pem = std::fs::read(&cfg.private_key_path)
            .with_context(|| format!("Cannot read key at {}", cfg.private_key_path))?;
        let rsa = Rsa::private_key_from_pem(&pem)
            .context("Failed to parse RSA private key")?;
        let pkey = PKey::from_rsa(rsa).context("Failed to wrap RSA key")?;
        Ok(Self { key_id: cfg.key_id.clone(), pkey })
    }

    /// Build the three Kalshi auth headers for a given HTTP method and full URL.
    /// Signing message: timestamp_ms_str + METHOD_UPPER + /trade-api/v2/path
    pub fn auth_headers(&self, method: &str, url: &str) -> Result<HashMap<String, String>> {
        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)?
            .as_millis();
        let ts_str = ts_ms.to_string();

        // Extract path-only (strip scheme + host + query string)
        let path = if url.starts_with("http") {
            let after_scheme = url.splitn(3, '/').nth(2).unwrap_or(url);
            // after_scheme is like "api.elections.kalshi.com/trade-api/v2/..."
            let path_start = after_scheme.find('/').map(|i| i).unwrap_or(0);
            after_scheme[path_start..].split('?').next().unwrap_or("")
        } else {
            url.split('?').next().unwrap_or(url)
        };

        let msg = format!("{}{}{}", ts_str, method.to_uppercase(), path);

        let mut signer = Signer::new(MessageDigest::sha256(), &self.pkey)?;
        // RSA-PSS with MGF1(SHA-256) and salt length = digest length (32 bytes)
        signer.set_rsa_padding(openssl::rsa::Padding::PKCS1_PSS)?;
        signer.set_rsa_mgf1_md(MessageDigest::sha256())?;
        signer.set_rsa_pss_saltlen(openssl::sign::RsaPssSaltlen::DIGEST_LENGTH)?;
        signer.update(msg.as_bytes())?;
        let sig = signer.sign_to_vec()?;

        let mut headers = HashMap::new();
        headers.insert("KALSHI-ACCESS-KEY".into(), self.key_id.clone());
        headers.insert("KALSHI-ACCESS-SIGNATURE".into(), B64.encode(&sig));
        headers.insert("KALSHI-ACCESS-TIMESTAMP".into(), ts_str);
        Ok(headers)
    }
}
