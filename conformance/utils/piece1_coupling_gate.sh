#!/usr/bin/env bash
# Falsifiable gate: prove a candidate PIECE-1 tree carries NO request-mode coupling.
#
# Piece 1 (peer-trait alignment + vendor registry + GUI + conformance tooling) must be
# correct ON ITS OWN, because it sits on main while piece 2 is still in review and is
# what an external implementor sees during that window. "I read the diff and it looked
# clean" is not evidence -- every defect this session survived exactly that check.
#
# Usage:  piece1_coupling_gate.sh [BASE]      (BASE defaults to origin/main)
# Exit 0 only if every check passes. Any failure prints what and why.

set -uo pipefail
BASE="${1:-origin/main}"
fail=0
note() { printf "  %-52s %s\n" "$1" "$2"; }
bad()  { fail=1; printf "  FAIL %-47s %s\n" "$1" "$2"; }

# 1. Request-mode symbols must not appear ANYWHERE in the base..HEAD diff.
#    Searching the diff, not the repo, is deliberate: the repo may legitimately
#    contain these names on main one day; what must stay clean is what THIS piece adds.
SYMS=(UnifiedParserStartingState UnifiedToolOutputMode UnifiedParserInit UnifiedPrompt
      UnifiedStartingStateInput initialize_request initialize_with_state
      initialize_with_output_mode initialize_from_prompt GuidedState GuidedJson)
# Exclude THIS script: it necessarily contains every forbidden symbol name, so
# scanning a diff that adds it makes the gate fail on itself.
diff_text="$(git diff "$BASE"...HEAD -- . ':(exclude)conformance/utils/piece1_coupling_gate.sh')"
for s in "${SYMS[@]}"; do
  n=$(printf '%s' "$diff_text" | grep -cE "^[+-].*\b${s}\b" || true)
  if [ "$n" -gt 0 ]; then bad "symbol '$s'" "appears $n time(s) in the diff"; fi
done
[ "$fail" -eq 0 ] && note "request-mode symbols in diff" "none (${#SYMS[@]} checked)"

# 2. Fixtures and the manifest must be untouched. Piece 1 owns no corpus data.
if git diff --exit-code --quiet "$BASE" -- conformance/fixtures conformance/fixtures-manifest.json; then
  note "conformance/fixtures + manifest" "byte-identical to $BASE"
else
  bad "conformance/fixtures + manifest" "MODIFIED — piece 1 must not touch corpus data"
fi

# 3. The two LFS pointers that pin the corpus must still be main's.
#    A pointer change is how a corpus transition leaks in without touching many files.
for f in conformance/fixtures/unified/golden.tar.gz conformance/fixtures/unified/inputs.tar.gz; do
  b=$(git show "$BASE:$f" 2>/dev/null | grep -oE 'oid sha256:[0-9a-f]+' | cut -d: -f2)
  h=$(git show "HEAD:$f"  2>/dev/null | grep -oE 'oid sha256:[0-9a-f]+' | cut -d: -f2)
  if [ -n "$b" ] && [ "$b" = "$h" ]; then note "$(basename "$f") pointer" "${b:0:16} (unchanged)"
  else bad "$(basename "$f") pointer" "${b:0:16} -> ${h:0:16}"; fi
done

# 4. The emitted corpus must still be main's 33 scenarios x 3 = 99 cases.
#    Counted from the GENERATOR, not the taxonomy map: the map reserves names the
#    generator does not emit, which is how a false 156-case denominator arose before.
if [ -f conformance/utils/src/gen_unified_golden.py ]; then
  # Written to a temp file rather than an inline heredoc: `$( <<EOF ... EOF || echo X )`
  # is a bash syntax error, and the check silently produced nothing until it was run
  # with output visible. A check that cannot fail loudly is not a check.
  _cnt_py=$(mktemp /tmp/piece1_count_XXXX.py)
  cat > "$_cnt_py" <<'PYEOF'
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("g", pathlib.Path("src/gen_unified_golden.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print(len(getattr(m, "CLEAN", [])) + len(getattr(m, "EDGE", [])))
PYEOF
  cnt=$( (cd conformance/utils && PYTHONPATH=src python3 "$_cnt_py") 2>/dev/null )
  rm -f "$_cnt_py"
  if [ "$cnt" = "33" ]; then note "generator scenarios" "33 (99 cases) — main's shape"
  elif [ -z "$cnt" ]; then bad "generator scenarios" "count FAILED to run — check manually"
  else bad "generator scenarios" "$cnt, expected 33 — corpus expansion belongs to piece 2"; fi
fi

# 5. The peer surface must be present and complete, with peer-shaped defaults.
M=parsers/v2/src/unified/mod.rs
if [ -f "$M" ]; then
  grep -q 'fn parse_into(&mut self, delta: &str, output: &mut UnifiedParserOutput) -> Result<()>;' "$M" \
    && note "parse_into" "required (no default)" || bad "parse_into" "not the required method"
  grep -q 'fn finish(&mut self) -> Result<UnifiedParserOutput>' "$M" \
    && note "finish" "returns UnifiedParserOutput" || bad "finish" "wrong return type"
  grep -qE 'fn initialize\(&mut self, _?prompt_token_ids: &\[u32\]\)' "$M" \
    && note "initialize(&[u32])" "peer signature present" || bad "initialize(&[u32])" "missing"
  grep -q 'Text(String)' "$M" && grep -q 'Reasoning(String)' "$M" \
    && note "UnifiedParserEvent" "peer tuple variants" || bad "UnifiedParserEvent" "not peer shape"
fi

echo
if [ "$fail" -eq 0 ]; then echo "  PIECE-1 COUPLING GATE: PASS"; else echo "  PIECE-1 COUPLING GATE: FAIL"; fi
exit "$fail"
