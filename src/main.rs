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
use std::{env, sync::Arc};
use tokio::process::Command;

// ---------------------------------------------------------------------
// Config / auth
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
// Tool AVAILABILITY is determined at runtime, not baked into the image:
// each ToolDef names the binary it depends on, and the agent checks
// `which <binary>` at startup. A tool whose binary isn't installed on
// this particular host simply doesn't appear in the agent card or
// tools/list, and calling it returns a clear "not available on this
// host" error rather than a confusing spawn failure. This lets one
// image run on every host with different tooling (installed via a
// per-host Brewfile at container start — see entrypoint.sh) without
// any per-host code or image changes.

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
    // Add new tools here as ToolDef entries. Pick whatever binary the
    // tool genuinely depends on for `requires_binary` — if that binary
    // isn't installed on a given host (no matching line in that host's
    // Brewfile), the tool just won't appear there. No code branching
    // per host needed.
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

async fn execute_tool(call: ToolCall) -> ToolResult {
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
            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {            let Some(service) = call.arguments.get("service").and_then(|v| v.as_str()) else {
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
            run_command(
                "docker",
                &["compose", "-f", COMPOSE_FILE, "restart", service],
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
            run_command("docker", &["ps", "--format", "{{.Names}}\t{{.Status}}"]).await
        }

        // Every id in TOOL_DEFS must have a matching arm above; this is
        // unreachable because execute_tool already looked the name up
        // in TOOL_DEFS before we get here.
        _ => unreachable!("tool id {} present in TOOL_DEFS without an execution arm", call.name),
    }
}

// Path to the compose file this agent is allowed to operate against.
// Bake this in at build time or override via env if you run multiple
// stacks; kept as a constant here to avoid the agent ever being told
// an arbitrary compose file path at call time.
const COMPOSE_FILE: &str = "/stack/compose.yaml";

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
        "description": "Host Docker/Anvil operations exposed as a narrow, allowlisted A2A tool surface, scoped to whatever tooling is actually installed on this host. Does not grant raw shell or docker-socket access to callers — every operation is a named, validated function.",
        "version": "0.2.0",
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
                .unwrap_or(true)                .unwrap_or(true)
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
            let result = execute_tool(call).await;
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

    let state = Arc::new(AppState {
        bearer_token: expected_token(),
        mesh_token: env::var("MESH_AUTH_TOKEN").ok(),
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
