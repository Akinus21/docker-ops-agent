// memory.rs
//
// Persisted store for what discovery found and what's been learned over
// time (via Hermes or manual correction), so the agent doesn't
// rediscover from scratch every restart and doesn't forget corrections.
// Lives in the mounted /data volume — content is host-specific runtime
// state, not code, so this stays consistent with the universal-image
// requirement the same way google-token.json already does for
// google-services-agent.

use crate::discovery::DiscoveredService;
use serde::{Deserialize, Serialize};
use tokio::fs;

#[derive(Debug, Serialize, Deserialize, Default, Clone)]
pub struct Memory {
    pub discovered_services: Vec<DiscoveredService>,
    pub learned_notes: Vec<LearnedNote>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct LearnedNote {
    pub note: String,
    pub source: String, // "hermes" | "manual"
    pub added: String,  // ISO 8601
}

const MEMORY_PATH: &str = "/data/memory.json";

pub async fn load() -> Memory {
    match fs::read_to_string(MEMORY_PATH).await {
        Ok(s) => serde_json::from_str(&s).unwrap_or_else(|e| {
            tracing::warn!("failed to parse {MEMORY_PATH}: {e} — starting with empty memory");
            Memory::default()
        }),
        Err(_) => {
            tracing::info!("no existing memory at {MEMORY_PATH} — starting fresh");
            Memory::default()
        }
    }
}

pub async fn save(mem: &Memory) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(mem).unwrap();
    fs::write(MEMORY_PATH, json).await
}

/// Replace this host's discovered-service list with a fresh discovery
/// run's results — services no longer found (renamed, removed) drop
/// out rather than lingering as stale entries pointing at nothing.
pub fn replace_discovered(mem: &mut Memory, fresh: Vec<DiscoveredService>) {
    mem.discovered_services = fresh;
}

pub fn find_service<'a>(mem: &'a Memory, role: &str) -> Option<&'a DiscoveredService> {
    mem.discovered_services.iter().find(|s| s.role == role)
}

pub fn add_note(mem: &mut Memory, note: String, source: &str) {
    mem.learned_notes.push(LearnedNote {
        note,
        source: source.to_string(),
        added: chrono::Utc::now().to_rfc3339(),
    });
}
