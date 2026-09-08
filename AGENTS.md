# frontend-crates agent instructions

- After any frontend-crates conformance fixture, parser, table-model, renderer, or conformance-documentation change, render the canonical report with `conformance/utils/render_table_v2.sh`, which writes `conformance/CONFORMANCE_v2.html`. Do this before reporting completion; do not stage or commit the generated HTML unless explicitly requested. After every render, report both its absolute filesystem path and its served `http://keivenc-linux1/dev/<worktree>/conformance/CONFORMANCE_v2.html` URL.
