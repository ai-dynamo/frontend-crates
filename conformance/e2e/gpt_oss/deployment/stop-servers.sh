#!/usr/bin/env bash
set -euo pipefail
for file in /work/server.pid /work/frontend-v1.pid /work/frontend-v2.pid /work/frontend-native-debug.pid; do
  if [ -f "$file" ]; then
    pid=$(cat "$file")
    if [[ "$pid" =~ ^[0-9]+$ ]]; then kill -TERM -- "-$pid" 2>/dev/null || true; fi
    rm "$file"
  fi
done
# Wait for children and listeners to exit before the next server binds the ports.
for attempt in $(seq 1 60); do
  if python3 - <<'PYPORT'
import socket
for port in (8000, 8001):
    with socket.socket() as connection:
        if connection.connect_ex(("127.0.0.1", port)) == 0:
            raise SystemExit(1)
PYPORT
  then exit 0; fi
  sleep 1
done
echo "Server ports did not close after termination" >&2
exit 1
