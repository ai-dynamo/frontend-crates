# dynamo-agent-rt-store

Durable persistence adapters for `dynamo-agent-rt`.

- DuckDB is an embedded, serialized, single-process store for local development, restart tests, and single-process deployments.
- PostgreSQL is the shared multi-replica production store.

The runtime core contains only persistence traits and has no database driver dependencies.
