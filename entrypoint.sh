#!/bin/bash
set -euo pipefail

# If this host's compose service mounts a Brewfile at /etc/docker-ops-agent/Brewfile,
# install whatever it lists before starting the agent. This is what makes one
# image serve every host with different tooling: the image never changes,
# only which Brewfile gets mounted per host.
BREWFILE="/etc/docker-ops-agent/Brewfile"

if [ -f "$BREWFILE" ]; then
    echo "[entrypoint] Found Brewfile at $BREWFILE — installing host-specific tools..."
    su - linuxbrew -c "brew bundle --file=$BREWFILE" || {
        echo "[entrypoint] WARNING: brew bundle failed — continuing with whatever installed successfully."
    }
else
    echo "[entrypoint] No Brewfile mounted at $BREWFILE — running with base image tooling only."
fi

exec /usr/local/bin/docker-ops-agent
