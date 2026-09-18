// discovery.rs
//
// Finds things the agent needs by inspecting the running host, guided
// by a per-host manifest (never hardcoded per-role logic) — mirrors the
// existing Brewfile pattern: one image, per-host mounted config decides
// what's actually relevant on that host. A host with no manifest mounted
// simply discovers nothing; the generic tools still work everywhere.

use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Debug, Deserialize, Clone)]
pub struct ServiceRoleConfig {
    pub role: String,
    pub image_hint: String,
    pub config_candidates: Vec<String>,
    #[serde(default)]
    pub reload_command: Vec<String>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DiscoveryManifest {
    #[serde(default)]
    pub services: Vec<ServiceRoleConfig>,
}

const MANIFEST_PATH: &str = "/etc/docker-ops-agent/discovery.toml";

pub async fn load_manifest() -> DiscoveryManifest {
    match tokio::fs::read_to_string(MANIFEST_PATH).await {
        Ok(s) => match toml::from_str(&s) {
            Ok(m) => m,
            Err(e) => {
                tracing::warn!("failed to parse {MANIFEST_PATH}: {e} — treating as empty manifest");
                DiscoveryManifest::default()
            }
        },
        Err(_) => {
            tracing::info!("no discovery manifest at {MANIFEST_PATH} — nothing to discover on this host");
            DiscoveryManifest::default()
        }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct DiscoveredService {
    pub role: String,
    pub container_name: String,
    pub config_path: Option<String>,
    pub reload_command: Vec<String>,
    pub last_confirmed: String, // ISO 8601
}

async fn find_container_by_image(image_hint: &str) -> Option<String> {
    let out = Command::new("docker")
        .args(["ps", "--format", "{{.Names}}	{{.Image}}"])
        .output()
        .await
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.lines().find_map(|line| {
        let mut parts = line.splitn(2, '	');
        let name = parts.next()?;
        let image = parts.next()?;
        image
            .to_lowercase()
            .contains(&image_hint.to_lowercase())
            .then(|| name.to_string())
    })
}

async fn find_config_path(container: &str, candidates: &[String]) -> Option<String> {
    for path in candidates {
        let ok = Command::new("docker")
            .args(["exec", container, "test", "-f", path])
            .status()
            .await
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            return Some(path.clone());
        }
    }
    None
}

/// Run discovery for every role in this host's manifest. Returns only
/// roles that were both declared AND actually found running — a role
/// declared in discovery.toml but not currently running is skipped
/// silently rather than producing a broken/empty entry.
pub async fn run_discovery() -> Vec<DiscoveredService> {
    let manifest = load_manifest().await;
    let mut discovered = vec![];
    let now = chrono::Utc::now().to_rfc3339();

    for svc in manifest.services {
        let Some(container_name) = find_container_by_image(&svc.image_hint).await else {
            tracing::info!(
                "discovery: role '{}' (image hint '{}') declared but not found running — skipping",
                svc.role, svc.image_hint
            );
            continue;
        };
        let config_path = find_config_path(&container_name, &svc.config_candidates).await;
        tracing::info!(
            "discovery: role '{}' -> container '{}', config_path {:?}",
            svc.role, container_name, config_path
        );
        discovered.push(DiscoveredService {
            role: svc.role,
            container_name,
            config_path,
            reload_command: svc.reload_command,
            last_confirmed: now.clone(),
        });
    }

    discovered
}
