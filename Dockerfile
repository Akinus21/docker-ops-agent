# Build stage — compile the Rust binary
FROM rust:1-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock* ./
COPY src ./src
RUN cargo build --release

# Runtime stage — Homebrew + Anvil live here since this is the container
# that actually needs to run `anvil docker update`, not Hermes.
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates docker-compose-plugin docker.io \
        build-essential procps curl file git sudo \
    && rm -rf /var/lib/apt/lists/*

# Homebrew refuses to run as root — dedicated non-root user owns the
# Homebrew install and runs anvil; the agent binary itself still runs
# as root below so it retains docker.sock access.
RUN useradd -m -s /bin/bash linuxbrew && \
    echo "linuxbrew ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER linuxbrew
WORKDIR /home/linuxbrew
RUN NONINTERACTIVE=1 /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
ENV PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:${PATH}"
# NOTE: no tools installed here — this image is identical across every
# host. Host-specific tooling (anvil, etc.) is installed at container
# start via a per-host Brewfile mounted at /etc/docker-ops-agent/Brewfile
# (see entrypoint.sh). Do not add `brew install <tool>` lines here —
# that would defeat the point of the one-image-per-host-tooling design.

USER root
# linuxbrew keeps NOPASSWD sudo permanently here (unlike the earlier
# single-purpose Hermes image) because `brew bundle` at container start
# may need to install system-level build dependencies for whatever a
# given host's Brewfile lists. This is an accepted, narrower tradeoff
# than mounting the docker socket into Hermes itself: the blast radius
# of this container being compromised is "this one host's docker/anvil
# operations," not "every host in the mesh."

ENV PATH="/home/linuxbrew/.linuxbrew/bin:/home/linuxbrew/.linuxbrew/sbin:${PATH}"

COPY --from=builder /build/target/release/docker-ops-agent /usr/local/bin/docker-ops-agent
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
