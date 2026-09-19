// docker-ops-agent
//
// A minimal A2A (Agent-to-Agent) server that exposes a small, explicit,
// allowlisted set of host operations (docker/anvil commands) to peers
// like Hermes — instead of giving the caller a raw docker socket or
// shell access.
//
// Design intent:
//   - Every operation is a named function with validated arguments,
//     not an arbitrary shell string. No command injection surface.
//   - Auth is a static bearer token (same pattern as opencode-a2a).
//   - The A2A surface is intentionally minimal: agent card discovery
//     + a single JSON-RPC endpoint that dispatches to the allowlist.
//   - Extend by adding a new arm to `execute_tool` + a `ToolDef` entry.
//   - The image itself is universal across every host — nothing
//     host-specific is hardcoded. Per-host tooling comes from a
//     mounted Brewfile (see entrypoint.sh); per-host service discovery
//     comes from a mounted discovery.toml (see discovery.rs). What's
//     actually found on a given host is persisted to /data/memory.json
//     and re-verified on demand via discover_environment.
 
mod discovery;
mod memory;
 
use axum::{
    extract::State,
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{collections::HashMap, env, sync::Arc};
use tokio::process::Command;
use tokio::sync::Mutex;
use uuid::Uuid;
 
// ---------------------------------------------------------------------
// Config / auth / shared state
// ---------------------------------------------------------------------
//
// Two distinct tokens are accepted:
//   - A2A_STATIC_AUTH_TOKEN: unique per host, used by Hermes to call
//     this specific agent. Never share this one across hosts.
//   - MESH_AUTH_TOKEN: one shared secret known to every docker-ops-agent
//     in the mesh, used ONLY for agent-to-agent calls (mesh_call).
//     Optional — if unset, this agent can still be called by Hermes,
//     it just can't participate in outbound mesh calls or accept them
//     from peers.
 
struct AppState {
    bearer_token: String,
    mesh_token: Option<String>,
    memory: Mutex<memory::Memory>,
    pending_proposals: Mutex<HashMap<String, ServiceProposal>>,
}
 
#[derive(Debug, Clone)]
struct ServiceProposal {
    role: String,
    content: String,
}
 
fn expected_token() -> String {
    env::var("A2A_STATIC_AUTH_TOKEN")
        .expect("A2A_STATIC_AUTH_TOKEN must be set — see README for format")
}
 
fn check_auth(headers: &HeaderMap, state: &AppState) -> bool {
    let Some(value) = headers.get("authorization") else {
        return false;
    };
    let Ok(value) = value.to_str() else {
        return false;
    };
    let Some(token) = value.strip_prefix("Bearer ") else {
        return false;
    };
    if token == state.bearer_token {
        return true;
    }
    if let Some(mesh_token) = &state.mesh_token {
        if token == mesh_token {
            return true;
        }
    }
    false
}
 
// ---------------------------------------------------------------------
// Allowlisted tools
// ---------------------------------------------------------------------
//
// Argument validation is deliberately strict: service/container names
// must match SAFE_NAME, numeric args are parsed as actual integers, and
// every tool builds its Command with fixed argv entries — never a
// shell string — so there is no injection path even if validation had
// a gap.
//
// Tool AVAILABILITY (binary-gated tools) is determined at runtime, not
// baked into the image: each ToolDef names the binary it depends on,
// and the agent checks `which <binary>` at startup. The service_config_*
// tools are additionally role-gated at call time against whatever this
// host's discovery actually found — see execute_tool.
 
static SAFE_NAME: Lazy<Regex> = Lazy::new(|| Regex::new(r"^[a-zA-Z0-9_.-]+$").unwrap());
 
fn valid_name(s: &str) -> bool {
    !s.is_empty() && s.len() <= 128 && SAFE_NAME.is_match(s)
}
 
struct ToolDef {
    id: &'static str,
    name: &'static str,
    description: &'static str,
    tags: &'static [&'static str],
    /// Binary this tool needs on $PATH. Checked once at startup via
    /// `which`. If absent, the tool is excluded from the agent card
    /// and tools/list, and calls to it are rejected up front.
    requires_binary: &'static str,
}
 
static TOOL_DEFS: &[ToolDef] = &[
    ToolDef {
        id: "anvil_docker_update",
        name: "Anvil Docker Update",
        description: "Run `anvil docker update --in-order <service>:<wait_seconds>`. Arguments: service (string, required), wait_seconds (integer, default 15, max 300).",
        tags: &["docker", "anvil", "deploy"],
        requires_binary: "anvil",
    },
    ToolDef {
        id: "compose_restart",
        name: "Compose Restart",
        description: "Restart a named service via `docker compose restart <service>` against the fixed stack compose file. Arguments: service (string, required).",
        tags: &["docker", "compose"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "docker_logs",
        name: "Docker Logs",
        description: "Fetch recent logs for a named container. Arguments: container (string, required), tail (integer, default 100, max 2000).",
        tags: &["docker", "debug"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "docker_ps",
        name: "Docker PS",
        description: "List running containers (name + status). No arguments.",
        tags: &["docker", "debug"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "network_diagnose",
        name: "Network Diagnose",
        description: "Diagnose DNS/connectivity for a hostname reachable from this host — runs getent hosts + docker network inspect against the bridge network. Arguments: target (string, required — a hostname or container name).",
        tags: &["network", "debug"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "discover_environment",
        name: "Discover Environment",
        description: "Re-run service discovery against this host's discovery.toml manifest (if mounted) and update memory with what's actually found running. Call this if a service was restarted/renamed and memory might be stale, or to check what roles this host actually has. No arguments.",
        tags: &["discovery", "memory"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "service_config_read",
        name: "Service Config Read",
        description: "Read the live config file for a discovered service role on this host. Arguments: role (string, required — must match a role this host's discovery manifest declares and actually found running; call discover_environment first if unsure what roles exist here).",
        tags: &["config", "debug"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "service_config_propose",
        name: "Service Config Propose",
        description: "Stage a proposed full replacement of a discovered service's config file without applying it. Returns a proposal_id and shows current vs proposed content side by side for review. Does NOT touch the live config. Arguments: role (string, required), proposed_content (string, required).",
        tags: &["config", "propose"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "service_config_apply",
        name: "Service Config Apply",
        description: "Apply a previously staged proposal and run that service role's reload command (if any). Only works with a valid, still-pending proposal_id — cannot apply arbitrary content directly. Requires explicit go-ahead from Gabriel in the requesting conversation before calling this, even with a valid proposal_id — see /data/instructions.md approval policy. Arguments: proposal_id (string, required).",
        tags: &["config", "apply"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "update_instructions",
        name: "Update Instructions",
        description: "Append a learned note to the human-editable instructions file at /data/instructions.md and to persisted memory. Use this to record something learned that should persist and guide future actions — a corrected path, a policy clarification Gabriel gave, etc. Arguments: note (string, required).",
        tags: &["memory", "instructions"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "container_restart",
        name: "Container Restart",
        description: "Restart a stopped or running container (docker restart). Note: this preserves the original env vars from when the container was first created — it does NOT re-read .env files. For containers whose env may have changed, use container_recreate instead.",
        tags: &["docker", "operate"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "container_recreate",
        name: "Container Recreate",
        description: "Recreate a container using the same image, ports, volumes, and environment as the original. The container is removed and recreated fresh, which forces Docker to re-read the current .env file for any env vars. Use this after editing .env to pick up changes that docker restart would miss. Arguments: container (string, required).",
        tags: &["docker", "operate"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "env_get",
        name: "Env Get",
        description: "Read environment variables from a running container as key=value lines. Arguments: container (string, required), filter (string, optional — only return vars whose name contains this substring, case-insensitive).",
        tags: &["docker", "debug", "env"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "env_update",
        name: "Env Update",
        description: "Update KEY=value entries in a .env file on the host filesystem (at /stack/.env by default, or in the directory specified by COMPOSE_DIR). Only modifies existing keys or adds new ones — never rewrites unrelated content. Returns a diff of what changed. Arguments: values (map of string->string, required — keys and their new values).",
        tags: &["docker", "compose", "env"],
        requires_binary: "gawk",
    },
    ToolDef {
        id: "compose_logs",
        name: "Compose Logs",
        description: "Fetch logs for a compose service from `docker compose logs`. Arguments: service (string, required), tail (integer, default 100, max 2000).",
        tags: &["docker", "compose", "debug"],
        requires_binary: "docker",
    },
    ToolDef {
        id: "compose_up_force_recreate",
        name: "Compose Up Force Recreate",
        description: "Run `docker compose up -d --force-recreate <service>` to stop, remove, and recreate a compose service with fresh env vars from the current .env file. Use this after editing .env to force Docker to re-read updated environment variables. Arguments: service (string, required).",
        tags: &["docker", "compose", "operate"],
        requires_binary: "docker",
    },
    // Add new tools here as ToolDef entries. Pick whatever binary the
    // tool genuinely depends on for `requires_binary` — if that binary
    // isn't installed on a given host (no matching line in that host's
    // Brewfile), the tool just won't appear there. No code branching
    // per host needed. For anything host-specific (a particular
    // service's config), prefer extending the generic service_config_*
    // tools' role handling rather than adding a new per-service tool —
    // that's what keeps the image universal.
];
 
/// Binaries found on $PATH at startup, checked once via `which`.
static AVAILABLE_BINARIES: Lazy<std::collections::HashSet<String>> = Lazy::new(detect_binaries);
 
fn detect_binaries() -> std::collections::HashSet<String> {
    let mut found = std::collections::HashSet::new();
    let mut needed: std::collections::HashSet<&str> =
        TOOL_DEFS.iter().map(|t| t.requires_binary).collect();
    needed.insert("docker"); // always worth knowing regardless of tool list
    for bin in needed {
        let ok = std::process::Command::new("which")
            .arg(bin)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if ok {
            found.insert(bin.to_string());
        }
    }
    found
}
 
fn available_tools() -> Vec<&'static ToolDef> {
    TOOL_DEFS
        .iter()
        .filter(|t| AVAILABLE_BINARIES.contains(t.requires_binary))
        .collect()
}
 
#[derive(Debug, Deserialize)]
struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}
 
#[derive(Debug, Serialize)]
struct ToolResult {
    ok: bool,
    output: String,
}
 
async fn run_command(program: &str, args: &[&str]) -> ToolResult {
    match Command::new(program).args(args).output().await {
        Ok(out) => {
            let mut combined = String::from_utf8_lossy(&out.stdout).to_string();
            combined.push_str(&String::from_utf8_lossy(&out.stderr));
            ToolResult {
                ok: out.status.success(),
                output: combined,
            }
        }
        Err(e) => ToolResult {
            ok: false,
            output: format!("failed to spawn `{program}`: {e}"),
        },
    }
}
 
// Path to the compose directory this agent is allowed to operate against.
// Override via COMPOSE_DIR env var; falls back to /stack for backwards
// compatibility. The directory must contain a compose.yaml or compose.yml.
const COMPOSE_DIR: &str = "/stack";
 
const INSTRUCTIONS_PATH: &str = "/data/instructions.md";
 
const DEFAULT_INSTRUCTIONS_TEMPLATE: &str = r#"# docker-ops-agent instructions
 
Rules this agent (and Hermes, when calling it) should follow when
executing gated actions. Hermes can append to this file via the
update_instructions tool; you can also edit it directly on disk —
this file is the single source of truth either way. Environment
details (which services live on this host, their config paths) are
tracked in /data/memory.json and discovery.toml, not here — this file
is for policy and learned notes, not raw facts.
 
## Approval policy
 
- service_config_apply requires a proposal_id from service_config_propose
  in the same session — the agent already refuses to apply without one.
- Do not run service_config_apply without an explicit go-ahead from
  Gabriel in the conversation that requested it, even if a proposal_id
  is valid. State the proposal is ready and wait for confirmation.
- compose_restart, docker_logs, docker_ps, network_diagnose, and
  service_config_read may be called freely — no approval needed, these
  are read-only or safely reversible.
 
## Notes learned over time
 
(Hermes appends dated entries here via update_instructions; manual
edits are equally valid — no special marker needed to distinguish them.)
"#;
 
async fn ensure_instructions_file_exists() {
    if tokio::fs::metadata(INSTRUCTIONS_PATH).await.is_err() {
        if let Err(e) = tokio::fs::write(INSTRUCTIONS_PATH, DEFAULT_INSTRUCTIONS_TEMPLATE).await {
            tracing::warn!("could not write default instructions template: {e}");
        } else {
            tracing::info!("wrote default instructions template to {INSTRUCTIONS_PATH}");
        }
    }
}
 
async fn execute_tool(state: &AppState, call: ToolCall) -> ToolResult {
    let Some(def) = TOOL_DEFS.iter().find(|t| t.id == call.name) else {
        return ToolResult {
            ok: false,
            output: format!(
                "unknown tool `{}` — see agent card `skills` for the allowlist",
                call.name
            ),
        };
    };
    if !AVAILABLE_BINARIES.contains(def.requires_binary) {
        return ToolResult {
            ok: false,
            output: format!(
                "tool `{}` requires `{}`, which is not installed on this host",
                def.id, def.requires_binary
            ),
        };
    }
 
    match call.name.as_str() {
        // anvil_docker_update { "service": "mcp-proxy", "wait_seconds": 15 }
        "anvil_docker_update" => {
            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `service` argument".into() };
            };
            if !valid_name(service) {
                return ToolResult { ok: false, output: "invalid `service` name".into() };
            }
            let wait = call
                .arguments
                .get("wait_seconds")
                .and_then(|v| v.as_u64())
                .unwrap_or(15)
                .min(300); // hard cap so a bad call can't hang forever
            let arg = format!("{service}:{wait}");
            run_command("anvil", &["docker", "update", "--in-order", &arg]).await
        }
 
        // compose_restart { "service": "mcp-proxy" }
        "compose_restart" => {
            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `service` argument".into() };
            };
            if !valid_name(service) {
                return ToolResult { ok: false, output: "invalid `service` name".into() };
            }
            let compose_dir = env::var("COMPOSE_DIR").unwrap_or_else(|_| COMPOSE_DIR.to_string());
            run_command(
                "docker",
                &["compose", "-f", &format!("{}/compose.yaml", compose_dir), "restart", service],
            )
            .await
        }
 
        // docker_logs { "container": "hermes", "tail": 100 }
        "docker_logs" => {
            let Some(container) = call.arguments.get("container").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `container` argument".into() };
            };
            if !valid_name(container) {
                return ToolResult { ok: false, output: "invalid `container` name".into() };
            }
            let tail = call
                .arguments
                .get("tail")
                .and_then(|v| v.as_u64())
                .unwrap_or(100)
                .min(2000);
            let tail_arg = tail.to_string();
            run_command("docker", &["logs", "--tail", &tail_arg, container]).await
        }
 
        // docker_ps {}
        "docker_ps" => {
            run_command("docker", &["ps", "--format", "{{.Names}}	{{.Status}}"]).await
        }
 
        // network_diagnose { "target": "opencode-a2a" }
        "network_diagnose" => {
            let Some(target) = call.arguments.get("target").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `target` argument".into() };
            };
            if !valid_name(target) {
                return ToolResult { ok: false, output: "invalid `target` name".into() };
            }
 
            let dns = run_command("getent", &["hosts", target]).await;
            let net = run_command(
                "docker",
                &["network", "inspect", "bridge", "--format", "{{json .Containers}}"],
            )
            .await;
 
            ToolResult {
                ok: dns.ok || net.ok,
                output: format!(
                    "--- getent hosts {target} ---
{}
--- docker network inspect (bridge) ---
{}",
                    dns.output, net.output
                ),
            }
        }
 
        // discover_environment {}
        "discover_environment" => {
            let fresh = discovery::run_discovery().await;
            let count = fresh.len();
            let summary: Vec<String> = fresh
                .iter()
                .map(|s| {
                    format!(
                        "{} -> container '{}', config_path {:?}",
                        s.role, s.container_name, s.config_path
                    )
                })
                .collect();
 
            let mut mem = state.memory.lock().await;
            memory::replace_discovered(&mut mem, fresh);
            let save_result = memory::save(&mem).await;
            drop(mem);
 
            if let Err(e) = save_result {
                return ToolResult {
                    ok: false,
                    output: format!("discovery ran ({count} services found) but failed to persist: {e}"),
                };
            }
 
            ToolResult {
                ok: true,
                output: if count == 0 {
                    "discovery ran — no services found (no discovery.toml mounted on this host, or none of its declared roles are currently running)".to_string()
                } else {
                    format!("discovery updated memory with {count} service(s):
{}", summary.join("
"))
                },
            }
        }
 
        // service_config_read { "role": "reverse_proxy" }
        "service_config_read" => {
            let Some(role) = call.arguments.get("role").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `role` argument".into() };
            };
            let mem = state.memory.lock().await;
            let Some(svc) = memory::find_service(&mem, role) else {
                return ToolResult {
                    ok: false,
                    output: format!(
                        "no discovered service for role `{role}` on this host — call discover_environment first, or check this host's discovery.toml"
                    ),
                };
            };
            let Some(config_path) = &svc.config_path else {
                return ToolResult {
                    ok: false,
                    output: format!("role `{role}` was discovered but no config path was found among its candidates"),
                };
            };
            let container = svc.container_name.clone();
            let config_path = config_path.clone();
            drop(mem);
 
            run_command("docker", &["exec", &container, "cat", &config_path]).await
        }
 
        // service_config_propose { "role": "reverse_proxy", "proposed_content": "..." }
        "service_config_propose" => {
            let Some(role) = call.arguments.get("role").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `role` argument".into() };
            };
            let Some(proposed) = call.arguments.get("proposed_content").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `proposed_content` argument".into() };
            };
 
            let mem = state.memory.lock().await;
            let Some(svc) = memory::find_service(&mem, role) else {
                return ToolResult {
                    ok: false,
                    output: format!("no discovered service for role `{role}` on this host — call discover_environment first"),
                };
            };
            let Some(config_path) = svc.config_path.clone() else {
                return ToolResult {
                    ok: false,
                    output: format!("role `{role}` has no known config path"),
                };
            };
            let container = svc.container_name.clone();
            drop(mem);
 
            let current = run_command("docker", &["exec", &container, "cat", &config_path]).await;
            if !current.ok {
                return ToolResult {
                    ok: false,
                    output: format!("could not read current config to compare against: {}", current.output),
                };
            }
 
            let proposal_id = Uuid::new_v4().to_string();
            state.pending_proposals.lock().await.insert(
                proposal_id.clone(),
                ServiceProposal { role: role.to_string(), content: proposed.to_string() },
            );
 
            ToolResult {
                ok: true,
                output: format!(
                    "proposal_id: {proposal_id}
role: {role}

--- CURRENT ---
{}

--- PROPOSED ---
{}

Review both, get Gabriel's explicit go-ahead per /data/instructions.md approval policy, then call service_config_apply with this proposal_id.",
                    current.output, proposed
                ),
            }
        }
 
        // service_config_apply { "proposal_id": "..." }
        "service_config_apply" => {
            let Some(proposal_id) = call.arguments.get("proposal_id").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `proposal_id` argument".into() };
            };
 
            let proposal = state.pending_proposals.lock().await.remove(proposal_id);
            let Some(proposal) = proposal else {
                return ToolResult {
                    ok: false,
                    output: "no pending proposal with that id — call service_config_propose first".into(),
                };
            };
 
            let mem = state.memory.lock().await;
            let Some(svc) = memory::find_service(&mem, &proposal.role) else {
                return ToolResult {
                    ok: false,
                    output: format!(
                        "role `{}` from this proposal is no longer known to this host (was memory cleared or the service removed?) — re-run discover_environment and start over",
                        proposal.role
                    ),
                };
            };
            let Some(config_path) = svc.config_path.clone() else {
                return ToolResult { ok: false, output: "service has no known config path".into() };
            };
            let container = svc.container_name.clone();
            let reload_command = svc.reload_command.clone();
            drop(mem);
 
            let staged_path = format!("{config_path}.new");
            let write = Command::new("docker")
                .args(["exec", "-i", &container, "sh", "-c", &format!("cat > {staged_path}")])
                .stdin(std::process::Stdio::piped())
                .spawn();
 
            let write_result = match write {
                Ok(mut child) => {
                    use tokio::io::AsyncWriteExt;
                    if let Some(mut stdin) = child.stdin.take() {
                        let _ = stdin.write_all(proposal.content.as_bytes()).await;
                    }
                    child.wait_with_output().await
                }
                Err(e) => {
                    return ToolResult { ok: false, output: format!("failed to write staged config: {e}") };
                }
            };
 
            match write_result {
                Ok(out) if out.status.success() => {}
                Ok(out) => {
                    return ToolResult {
                        ok: false,
                        output: format!("staged write failed: {}", String::from_utf8_lossy(&out.stderr)),
                    };
                }
                Err(e) => {
                    return ToolResult { ok: false, output: format!("staged write failed: {e}") };
                }
            }
 
            let mv = run_command("docker", &["exec", &container, "mv", &staged_path, &config_path]).await;
            if !mv.ok {
                return ToolResult { ok: false, output: format!("failed to move staged config into place: {}", mv.output) };
            }
 
            if reload_command.is_empty() {
                return ToolResult {
                    ok: true,
                    output: "config written and moved into place. No reload_command configured for this role — the service may need a manual restart (see compose_restart).".to_string(),
                };
            }
 
            // Substitute {config_path} in any reload_command arg with the
            // real path, then run it inside the target container.
            let substituted: Vec<String> = reload_command
                .iter()
                .map(|arg| arg.replace("{config_path}", &config_path))
                .collect();
            let mut exec_args: Vec<&str> = vec!["exec", &container];
            exec_args.extend(substituted.iter().map(|s| s.as_str()));
 
            let reload = run_command("docker", &exec_args).await;
            ToolResult {
                ok: reload.ok,
                output: format!("config written and moved into place.
reload output: {}", reload.output),
            }
        }
 
        // update_instructions { "note": "..." }
        "update_instructions" => {
            let Some(note) = call.arguments.get("note").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `note` argument".into() };
            };
 
            let timestamp = chrono::Utc::now().to_rfc3339();
            let entry = format!("
- [{timestamp}] {note}");
            let file_result = tokio::fs::OpenOptions::new()
                .append(true)
                .open(INSTRUCTIONS_PATH)
                .await;
 
            let file_ok = match file_result {
                Ok(mut file) => {
                    use tokio::io::AsyncWriteExt;
                    file.write_all(entry.as_bytes()).await.is_ok()
                }
                Err(_) => false,
            };
 
            let mut mem = state.memory.lock().await;
            memory::add_note(&mut mem, note.to_string(), "hermes");
            let save_result = memory::save(&mem).await;
            drop(mem);
 
            ToolResult {
                ok: file_ok && save_result.is_ok(),
                output: format!(
                    "instructions.md append: {}, memory.json save: {}",
                    if file_ok { "ok" } else { "failed" },
                    if save_result.is_ok() { "ok" } else { "failed" }
                ),
            }
        }
 
        // container_restart { "container": "vikunja" }
        "container_restart" => {
            let Some(container) = call.arguments.get("container").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `container` argument".into() };
            };
            if !valid_name(container) {
                return ToolResult { ok: false, output: "invalid `container` name".into() };
            }
            run_command("docker", &["restart", container]).await
        }

        // container_recreate { "container": "vikunja" }
        // Stops the container, removes it, then re-creates it fresh using
        // the same image/ports/volumes but with the current .env file's
        // env vars as the base. Extra vars from the original container's
        // docker-created env (not from .env) are preserved.
        "container_recreate" => {
            let Some(container) = call.arguments.get("container").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `container` argument".into() };
            };
            if !valid_name(container) {
                return ToolResult { ok: false, output: "invalid `container` name".into() };
            }

            // 1. Inspect the container's full config (including HostConfig which moved
            // out of Config in newer Docker API versions)
            let inspect = run_command(
                "docker",
                &["inspect", "--format", "{{json .}}", container],
            )
            .await;
            if !inspect.ok {
                return ToolResult {
                    ok: false,
                    output: format!("failed to inspect container: {}", inspect.output),
                };
            }

            #[derive(serde::Deserialize)]
            struct ContainerInspect {
                #[serde(rename = "Config")]
                config: ContainerConfig,
            }

            #[derive(serde::Deserialize)]
            struct ContainerConfig {
                #[serde(rename = "Env")]
                env: Vec<String>,
                #[serde(rename = "Image")]
                image: String,
                #[serde(rename = "ExposedPorts", default)]
                #[allow(dead_code)]
                exposed_ports: Option<serde_json::Value>,
                #[serde(rename = "HostConfig")]
                host_config: serde_json::Value,
            }

            // docker inspect returns a single object {Config: {...}, HostConfig: {...}, ...}
            let inspect_obj: ContainerInspect = match serde_json::from_str(&inspect.output) {
                Ok(c) => c,
                Err(e) => {
                    return ToolResult {
                        ok: false,
                        output: format!("failed to parse docker inspect JSON: {e} — output was: {}", inspect.output),
                    };
                }
            };
            let cfg = inspect_obj.config;

            // 2. Stop the container
            let stop = run_command("docker", &["stop", container]).await;
            if !stop.ok {
                return ToolResult {
                    ok: false,
                    output: format!("stop failed: {}", stop.output),
                };
            }

            // 3. Remove the container
            let rm = run_command("docker", &["rm", container]).await;
            if !rm.ok {
                return ToolResult {
                    ok: false,
                    output: format!("rm failed (container may still be present): {}", rm.output),
                };
            }

            // 4. Re-create: build a new docker run command
            // Read current .env for fresh env vars
            let compose_dir = env::var("COMPOSE_DIR").unwrap_or_else(|_| COMPOSE_DIR.to_string());
            let env_path = format!("{}/.env", compose_dir);
            let current_env: std::collections::HashMap<String, String> =
                tokio::fs::read_to_string(&env_path)
                    .await
                    .map(|content| {
                        content
                            .lines()
                            .filter_map(|line| {
                                let line = line.trim();
                                if line.is_empty() || line.starts_with('#') {
                                    return None;
                                }
                                line.split_once('=').map(|(k, v)| {
                                    (k.to_string(), v.to_string())
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

            // Start with current .env vars, overlay the original container's env
            // vars (which may include DOCKER_-created entries not in .env)
            let current_env_count = current_env.len();
            let mut final_env: std::collections::HashMap<String, String> = current_env;
            for env_line in &cfg.env {
                if let Some((k, v)) = env_line.split_once('=') {
                    final_env.insert(k.to_string(), v.to_string());
                }
            }

            // Build args: docker run [image]
            let mut args: Vec<String> = vec!["run".to_string(), "-d".to_string(), "--name".to_string(), container.to_string()];

            // Re-attach ports from HostConfig.PortBindings
            if let Some(port_bindings) = cfg
                .host_config
                .get("PortBindings")
                .and_then(|pb| pb.as_object())
            {
                for (container_port, host_binding) in port_bindings {
                    if let Some(bindings) = host_binding.as_array() {
                        for binding in bindings {
                            let _host_ip = binding
                                .get("HostIp")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0.0.0.0");
                            let host_port = binding
                                .get("HostPort")
                                .and_then(|v| v.as_str())
                                .unwrap_or("0");
                            args.push(format!("-p{}:{}", host_port, container_port));
                        }
                    }
                }
            }

            // Re-attach volumes from HostConfig.Binds
            if let Some(binds) = cfg
                .host_config
                .get("Binds")
                .and_then(|b| b.as_array())
            {
                for bind in binds {
                    if let Some(vol) = bind.as_str() {
                        args.push(format!("-v={}", vol));
                    }
                }
            }

            // Add env vars
            for (key, val) in &final_env {
                args.push(format!("-e{}={}", key, val));
            }

            args.push(cfg.image.clone());

            let run_result = Command::new("docker")
                .args(&args)
                .output()
                .await;

            match run_result {
                Ok(out) if out.status.success() => {
                    ToolResult {
                        ok: true,
                        output: format!(
                            "container recreated successfully.\n\
                            Image: {}\n\
                            Env vars applied from {} ({} vars from .env, {} extra from original container)\n\
                            Run command: docker {}\n\
                            Container output: {}",
                            cfg.image,
                            env_path,
                            current_env_count,
                            cfg.env.len() - current_env_count,
                            args.join(" "),
                            String::from_utf8_lossy(&out.stdout).trim()
                        ),
                    }
                }
                Ok(out) => ToolResult {
                    ok: false,
                    output: format!(
                        "container stopped and removed but failed to re-create:\n{}\n\
                        Container is now gone — you need to manually run:\n\
                        docker {}\n\
                        Error: {}",
                        String::from_utf8_lossy(&out.stderr),
                        args.join(" "),
                        String::from_utf8_lossy(&out.stdout)
                    ),
                },
                Err(e) => ToolResult {
                    ok: false,
                    output: format!(
                        "container stopped and removed but failed to spawn docker run: {e}\n\
                        You need to manually re-create the container."
                    ),
                },
            }
        }

        // env_get { "container": "vikunja", "filter": "VIKUNJA" }
        "env_get" => {
            let Some(container) = call.arguments.get("container").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `container` argument".into() };
            };
            if !valid_name(container) {
                return ToolResult { ok: false, output: "invalid `container` name".into() };
            }
            let filter = call
                .arguments
                .get("filter")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let out = run_command("docker", &["exec", container, "env"]).await;
            let vars = &out.output;
            let filtered: Vec<&str> = if filter.is_empty() {
                vars.lines().collect()
            } else {
                vars.lines()
                    .filter(|l| l.contains(filter))
                    .collect()
            };
            ToolResult {
                ok: out.ok,
                output: filtered.join("\n"),
            }
        }

        // env_update { "values": { "VIKUNJA_SERVICE_PUBLICURL": "https://todo.akinus21.com" } }
        "env_update" => {
            let Some(values) = call.arguments.get("values").and_then(|v| v.as_object()) else {
                return ToolResult { ok: false, output: "missing `values` argument (must be a map)".into() };
            };
            if values.is_empty() {
                return ToolResult { ok: false, output: "`values` map is empty — nothing to update".into() };
            }
            let compose_dir = env::var("COMPOSE_DIR").unwrap_or_else(|_| COMPOSE_DIR.to_string());
            let env_path = format!("{}/.env", compose_dir);

            // Read current env file
            let current = match tokio::fs::read_to_string(&env_path).await {
                Ok(c) => c,
                Err(e) => {
                    return ToolResult {
                        ok: false,
                        output: format!("could not read {}: {e}", env_path),
                    };
                }
            };

            let mut changes = vec![];
            let mut new_lines = Vec::new();
            let mut updated_keys = std::collections::HashSet::new();

            for line in current.lines() {
                let trimmed = line.trim();
                if trimmed.is_empty() || trimmed.starts_with('#') {
                    new_lines.push(line.to_string());
                    continue;
                }
                if let Some((key, _)) = line.split_once('=') {
                    if let Some(new_val) = values.get(key) {
                        new_lines.push(format!("{}={}", key, new_val));
                        changes.push(format!("  {}: (was {:?}) -> {:?}", key, line.split_once('=').map(|(_, v)| v), new_val));
                        updated_keys.insert(key.to_string());
                        continue;
                    }
                }
                new_lines.push(line.to_string());
            }
            // Add keys that weren't in the file
            for (key, new_val) in values {
                if !updated_keys.contains(key) {
                    new_lines.push(format!("{}={}", key, new_val));
                    changes.push(format!("  {}: (new) -> {:?}", key, new_val));
                }
            }

            let new_content = new_lines.join("\n");
            if let Err(e) = tokio::fs::write(&env_path, new_content).await {
                return ToolResult {
                    ok: false,
                    output: format!("failed to write {}: {e}", env_path),
                };
            }

            ToolResult {
                ok: true,
                output: if changes.is_empty() {
                    "no changes needed — all keys already have the requested values".into()
                } else {
                    format!("updated {}:\n{}", env_path, changes.join("\n"))
                },
            }
        }

        // compose_logs { "service": "vikunja", "tail": 100 }
        "compose_logs" => {
            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `service` argument".into() };
            };
            if !valid_name(service) {
                return ToolResult { ok: false, output: "invalid `service` name".into() };
            }
            let tail = call
                .arguments
                .get("tail")
                .and_then(|v| v.as_u64())
                .unwrap_or(100)
                .min(2000);
            let compose_dir = env::var("COMPOSE_DIR").unwrap_or_else(|_| COMPOSE_DIR.to_string());
            run_command(
                "docker",
                &[
                    "compose",
                    "-f",
                    &format!("{}/compose.yaml", compose_dir),
                    "logs",
                    "--tail",
                    &tail.to_string(),
                    service,
                ],
            )
            .await
        }

        // compose_up_force_recreate { "service": "vikunja" }
        "compose_up_force_recreate" => {
            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {
                return ToolResult { ok: false, output: "missing `service` argument".into() };
            };
            if !valid_name(service) {
                return ToolResult { ok: false, output: "invalid `service` name".into() };
            }
            let compose_dir = env::var("COMPOSE_DIR").unwrap_or_else(|_| COMPOSE_DIR.to_string());
            run_command(
                "docker",
                &[
                    "compose",
                    "-f",
                    &format!("{}/compose.yaml", compose_dir),
                    "up",
                    "-d",
                    "--force-recreate",
                    service,
                ],
            )
            .await
        }

        // Every id in TOOL_DEFS must have a matching arm above; this is
        // unreachable because execute_tool already looked the name up
        // in TOOL_DEFS before we get here.
        _ => unreachable!("tool id {} present in TOOL_DEFS without an execution arm", call.name),
    }
}
 
// ---------------------------------------------------------------------
// A2A surface: agent card + JSON-RPC endpoint
// ---------------------------------------------------------------------
 
async fn agent_card() -> Json<Value> {
    let host = env::var("A2A_HOST_LABEL").unwrap_or_else(|_| "unknown-host".to_string());
    let self_card_url = env::var("A2A_SELF_CARD_URL")
        .unwrap_or_else(|_| "http://docker-ops-agent:8000/.well-known/agent-card.json".to_string());
    let self_rpc_base = peer_rpc_base(&self_card_url);
    let skills: Vec<Value> = available_tools()
        .iter()
        .map(|t| {
            json!({
                "id": t.id,
                "name": t.name,
                "description": t.description,
                "tags": t.tags
            })
        })
        .collect();
 
    Json(json!({
        "name": format!("docker-ops-agent ({host})"),
        "description": "Host Docker/Anvil operations exposed as a narrow, allowlisted A2A tool surface, scoped to whatever tooling is actually installed on this host and whatever services this host's discovery.toml manifest declares. Does not grant raw shell or docker-socket access to callers — every operation is a named, validated function. Config-mutating tools (service_config_apply) require staging a reviewable proposal first.",
        "version": "0.5.0",
        "hostLabel": host,
        "supportedInterfaces": [
            { "url": self_rpc_base, "protocolBinding": "JSONRPC", "protocolVersion": "1.0" }
        ],
        "capabilities": { "streaming": false },
        "securitySchemes": {
            "bearerAuth": {
                "httpAuthSecurityScheme": {
                    "description": "Bearer token authentication",
                    "scheme": "bearer",
                    "bearerFormat": "opaque"
                }
            }
        },
        "securityRequirements": [ { "schemes": { "bearerAuth": {} } } ],
        "defaultInputModes": ["application/json"],
        "defaultOutputModes": ["application/json"],
        "skills": skills
    }))
}
 
// ---------------------------------------------------------------------
// Outbound mesh calling — this is what makes the mesh a real mesh
// rather than hub-and-spoke: any docker-ops-agent can query the
// registry itself and call another peer directly, without routing
// through Hermes. Uses curl (already shelled out to elsewhere in this
// file) rather than adding an HTTP client dependency.
// ---------------------------------------------------------------------
 
#[derive(Debug, Deserialize, Serialize, Clone)]
struct RegistryPeer {
    host_label: String,
    role: String,
    agent_card_url: String,
}
 
async fn fetch_registry_peers() -> Result<Vec<RegistryPeer>, String> {
    let registry_url = env::var("REGISTRY_URL")
        .map_err(|_| "REGISTRY_URL is not configured on this agent".to_string())?;
    let out = Command::new("curl")
        .args(["-s", &format!("{registry_url}/peers")])
        .output()
        .await
        .map_err(|e| format!("failed to reach registry: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "registry request exited non-zero: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let body: Value = serde_json::from_slice(&out.stdout)
        .map_err(|e| format!("registry returned invalid JSON: {e}"))?;
    let peers: Vec<RegistryPeer> = serde_json::from_value(
        body.get("peers").cloned().unwrap_or(json!([])),
    )
    .map_err(|e| format!("could not parse peer list: {e}"))?;
    Ok(peers)
}
 
/// Derive the base JSON-RPC URL (the POST / endpoint) from a peer's
/// published agent-card URL, which always ends in
/// /.well-known/agent-card.json per the A2A spec.
fn peer_rpc_base(agent_card_url: &str) -> String {
    agent_card_url
        .trim_end_matches("/.well-known/agent-card.json")
        .to_string()
}
 
/// Call another agent's tools/call endpoint using the shared mesh
/// token. Returns the raw JSON result the peer sent back.
async fn call_peer(
    rpc_base: &str,
    mesh_token: &str,
    tool_name: &str,
    arguments: Value,
) -> Result<Value, String> {
    let payload = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": { "name": tool_name, "arguments": arguments }
    })
    .to_string();
 
    let out = Command::new("curl")
        .args([
            "-s",
            "-X",
            "POST",
            rpc_base,
            "-H",
            &format!("Authorization: Bearer {mesh_token}"),
            "-H",
            "Content-Type: application/json",
            "-d",
            &payload,
        ])
        .output()
        .await
        .map_err(|e| format!("failed to reach peer: {e}"))?;
 
    if !out.status.success() {
        return Err(format!(
            "peer request exited non-zero: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    serde_json::from_slice(&out.stdout).map_err(|e| format!("peer returned invalid JSON: {e}"))
}
 
#[derive(Debug, Deserialize)]
struct MeshCallParams {
    /// Match a peer by host_label (preferred — unambiguous) or, if
    /// omitted, by role (only safe when exactly one peer has that role).
    host_label: Option<String>,
    role: Option<String>,
    tool_name: String,
    #[serde(default)]
    arguments: Value,
}
 
async fn handle_mesh_call(state: &AppState, params: Value) -> Result<Value, String> {
    let mesh_token = state.mesh_token.clone().ok_or_else(|| {
        "MESH_AUTH_TOKEN is not configured on this agent — cannot make outbound mesh calls"
            .to_string()
    })?;
 
    let call: MeshCallParams =
        serde_json::from_value(params).map_err(|e| format!("invalid mesh_call params: {e}"))?;
 
    let peers = fetch_registry_peers().await?;
 
    let matches: Vec<&RegistryPeer> = peers
        .iter()
        .filter(|p| {
            call.host_label
                .as_ref()
                .map(|h| &p.host_label == h)
                .unwrap_or(true)
                && call.role.as_ref().map(|r| &p.role == r).unwrap_or(true)
        })
        .collect();
 
    let target = match matches.len() {
        0 => return Err("no registered peer matches the given host_label/role".to_string()),
        1 => matches[0],
        _ => {
            return Err(format!(
                "{} peers match — specify host_label to disambiguate",
                matches.len()
            ))
        }
    };
 
    let rpc_base = peer_rpc_base(&target.agent_card_url);
    call_peer(&rpc_base, &mesh_token, &call.tool_name, call.arguments).await
}
 
#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}
 
fn jsonrpc_error(id: Option<Value>, code: i64, message: &str) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message }
    }))
}
 
fn jsonrpc_result(id: Option<Value>, result: Value) -> Json<Value> {
    Json(json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    }))
}
 
async fn rpc_handler(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    if !check_auth(&headers, &state) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({ "error": "missing or invalid bearer token" })),
        )
            .into_response();
    }
 
    let req: JsonRpcRequest = match serde_json::from_slice(&body) {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_REQUEST,
                jsonrpc_error(None, -32700, &format!("parse error: {e}")),
            )
                .into_response();
        }
    };
 
    match req.method.as_str() {
        // Single dispatch method: { "method": "tools/call", "params": { "name": "...", "arguments": {...} } }
        "tools/call" => {
            let call: ToolCall = match serde_json::from_value(req.params) {
                Ok(c) => c,
                Err(e) => {
                    return (
                        StatusCode::OK,
                        jsonrpc_error(req.id, -32602, &format!("invalid params: {e}")),
                    )
                        .into_response();
                }
            };
            let result = execute_tool(&state, call).await;
            (
                StatusCode::OK,
                jsonrpc_result(req.id, json!({ "ok": result.ok, "output": result.output })),
            )
                .into_response()
        }
        "tools/list" => {
            let card = agent_card().await;
            let skills = card.0.get("skills").cloned().unwrap_or(json!([]));
            (StatusCode::OK, jsonrpc_result(req.id, json!({ "tools": skills }))).into_response()
        }
        // mesh_list_peers: returns whatever the registry currently has
        // (no auth beyond this agent's own bearer/mesh token already
        // checked above — the registry's own GET /peers is unauthed by
        // design, this just proxies it for convenience).
        "mesh_list_peers" => match fetch_registry_peers().await {
            Ok(peers) => (StatusCode::OK, jsonrpc_result(req.id, json!({ "peers": peers })))
                .into_response(),
            Err(e) => (StatusCode::OK, jsonrpc_error(req.id, -32000, &e)).into_response(),
        },
        // mesh_call: forward a tools/call to another peer, found via
        // host_label/role in the registry. This is the actual mesh —
        // any agent can reach any other agent directly.
        "mesh_call" => match handle_mesh_call(&state, req.params).await {
            Ok(result) => (StatusCode::OK, jsonrpc_result(req.id, result)).into_response(),
            Err(e) => (StatusCode::OK, jsonrpc_error(req.id, -32001, &e)).into_response(),
        },
        // Simple liveness/ping used during initial peer setup (matches
        // the informal "Pong!" convention seen in opencode-a2a testing).
        "ping" => (StatusCode::OK, jsonrpc_result(req.id, json!("Pong!"))).into_response(),
        other => (
            StatusCode::OK,
            jsonrpc_error(req.id, -32601, &format!("method not found: {other}")),
        )
            .into_response(),
    }
}
 
async fn agent_card_route() -> impl IntoResponse {
    (StatusCode::OK, agent_card().await)
}
 
// ---------------------------------------------------------------------
// Mesh registry self-registration (stand-in for DNS-SD advertising —
// see README for why). If REGISTRY_URL is set, this agent periodically
// announces its own agent-card URL, host label, and role so Hermes can
// discover it automatically instead of needing a hardcoded URL per host.
// Silently does nothing if REGISTRY_URL isn't set — registration is
// optional, the agent works standalone without it.
// ---------------------------------------------------------------------
 
async fn registry_heartbeat_loop() {
    let Ok(registry_url) = env::var("REGISTRY_URL") else {
        tracing::info!("REGISTRY_URL not set — mesh registry self-registration disabled");
        return;
    };
    let Ok(registry_token) = env::var("REGISTRY_TOKEN") else {
        tracing::warn!("REGISTRY_URL set but REGISTRY_TOKEN missing — cannot self-register");
        return;
    };
    let host_label = env::var("A2A_HOST_LABEL").unwrap_or_else(|_| "unknown-host".to_string());
    let self_url = env::var("A2A_SELF_CARD_URL").unwrap_or_else(|_| {
        "http://docker-ops-agent:8000/.well-known/agent-card.json".to_string()
    });
    let ttl_seconds: u64 = 90;
    let heartbeat_every = std::time::Duration::from_secs(30);
 
    let payload = json!({
        "host_label": host_label,
        "role": "docker-ops",
        "agent_card_url": self_url,
        "ttl_seconds": ttl_seconds
    })
    .to_string();
 
    loop {
        let result = Command::new("curl")
            .args([
                "-s",
                "-o",
                "/dev/null",
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                &format!("{registry_url}/register"),
                "-H",
                &format!("X-Registry-Token: {registry_token}"),
                "-H",
                "Content-Type: application/json",
                "-d",
                &payload,
            ])
            .output()
            .await;
 
        match result {
            Ok(out) => {
                let code = String::from_utf8_lossy(&out.stdout).to_string();
                if code == "200" {
                    tracing::debug!("registry heartbeat ok");
                } else {
                    tracing::warn!("registry heartbeat returned HTTP {code}");
                }
            }
            Err(e) => tracing::warn!("registry heartbeat failed to run curl: {e}"),
        }
 
        tokio::time::sleep(heartbeat_every).await;
    }
}
 
#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();
 
    ensure_instructions_file_exists().await;
 
    // Startup discovery: find whatever this host's discovery.toml (if
    // any) declares and is actually running, and persist it. A host
    // with no manifest mounted just ends up with an empty
    // discovered_services list — not an error.
    let mut mem = memory::load().await;
    let fresh = discovery::run_discovery().await;
    tracing::info!("startup discovery found {} service(s)", fresh.len());
    memory::replace_discovered(&mut mem, fresh);
    if let Err(e) = memory::save(&mem).await {
        tracing::warn!("failed to persist startup discovery results: {e}");
    }
 
    let state = Arc::new(AppState {
        bearer_token: expected_token(),
        mesh_token: env::var("MESH_AUTH_TOKEN").ok(),
        memory: Mutex::new(mem),
        pending_proposals: Mutex::new(HashMap::new()),
    });
 
    let app = Router::new()
        .route("/.well-known/agent-card.json", get(agent_card_route))
        .route("/", post(rpc_handler))
        .with_state(state);
 
    tokio::spawn(registry_heartbeat_loop());
 
    let port = env::var("BIND_PORT").unwrap_or_else(|_| "8000".to_string());
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("docker-ops-agent listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
