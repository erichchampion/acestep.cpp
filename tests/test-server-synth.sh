#!/usr/bin/env bash
# test-server-synth.sh: ace-server's /synth refusals, against a running server
# (cadenza-audio #1194, #1286). Starts ace-server on a free port of 127.0.0.1
# with the test models, POSTs each request /synth must refuse before any job
# -- a plain-JSON cover with no source, a multipart cover with only its
# request part, and a cover with a ref_audio part and no source -- and
# requires a 400 that says why, and that the server still answers after
# each. The server is stopped however the test ends.
#
# And (cadenza-audio #1170): a multipart of a task the engine does not know,
# with an audio part that is no audio, is refused by the batch rule, by name
# -- one check, before any part is read, not "Failed to decode audio"; and a
# text2music sent such a part gets its job, the part unread, and the log's
# warning names that job by its id, in the engine's words. That job is
# cancelled at once.
#
# The first draft of #1193 refused that last cover after decoding its
# reference, and freed the reference twice. The refusal now comes before any
# part is read, so this holds that it does -- the reference never decoded --
# not the free itself: no request reaches the later refusal with a decoded
# reference (a source part that decodes to nothing is a 400 of its own,
# "Failed to decode audio"), which stays as a guard.
#
#   test-server-synth.sh <ace-server> <models-dir> <ref.wav>
set -euo pipefail

if [ $# -ne 3 ]; then
    echo "usage: $0 <ace-server> <models-dir> <ref.wav>" >&2
    exit 2
fi
SERVER="$1"
MODELS="$2"
REF="$3"
# How long the server may take to scan the models and bind, and a request to
# be answered: none loads a model, so both are short.
START_TIMEOUT_S=60
REQUEST_TIMEOUT_S=30
# The engine's reason for a task that works from a source given none, and the
# hint each path adds to it (tools/ace-server.cpp's kSourceHint*), as the 400's
# JSON escapes them.
REASON='the task requires source audio or latents, and none was given'
HINT_JSON='(send it as a multipart request, with an \"audio\" or \"src_latents\" part)'
HINT_MULTIPART='(part \"audio\" or \"src_latents\")'
COVER='{"task_type":"cover","caption":"warm analog synth","duration":4}'

work="$(mktemp -d)"
pid=""
stop_server() {
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 50); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 0.1
        done
        kill -KILL "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi
    rm -rf "$work"
}
trap stop_server EXIT
trap 'exit 1' INT TERM

fail() {
    echo "FAIL: $*" >&2
    echo "--- server log ---" >&2
    cat "$work/server.log" >&2 || true
    exit 1
}

"$SERVER" --models "$MODELS" --host 127.0.0.1 --port 0 2>"$work/server.log" &
pid=$!

port=""
for _ in $(seq 1 $((START_TIMEOUT_S * 10))); do
    kill -0 "$pid" 2>/dev/null || fail "ace-server exited before it listened"
    port="$(sed -n 's/^\[Server\] Listening on 127\.0\.0\.1:\([0-9][0-9]*\)$/\1/p' "$work/server.log" | head -n 1)"
    [ -z "$port" ] || break
    sleep 0.1
done
[ -n "$port" ] || fail "ace-server did not listen within ${START_TIMEOUT_S} s"
URL="http://127.0.0.1:$port"

# The server answers, and is the one this test started.
answers() {
    local body
    body="$(curl -sS --max-time "$REQUEST_TIMEOUT_S" "$URL/health")" || fail "no answer from /health $1"
    [ "$body" = '{"status":"ok"}' ] || fail "/health $1 said: $body"
    kill -0 "$pid" 2>/dev/null || fail "ace-server is gone $1"
}

# POST /synth with curl's remaining arguments: a 400 saying the reason and
# `hint`, and the server still answering.
refused() {
    local name="$1" hint="$2" code
    shift 2
    code="$(curl -sS --max-time "$REQUEST_TIMEOUT_S" -o "$work/body" -w '%{http_code}' "$@" "$URL/synth")" ||
        fail "$name: no answer"
    [ "$code" = 400 ] || fail "$name: HTTP $code, want 400: $(cat "$work/body")"
    grep -qF "$REASON" "$work/body" || fail "$name: not the engine's reason: $(cat "$work/body")"
    grep -qF "$hint" "$work/body" || fail "$name: not the hint $hint: $(cat "$work/body")"
    answers "after $name"
    echo "ok: $name -- $(cat "$work/body")"
}

answers "at the start"
refused "a plain-JSON cover with no source" "$HINT_JSON" \
    -H 'Content-Type: application/json' --data-binary "$COVER"
refused "a multipart cover with only its request part" "$HINT_MULTIPART" \
    --form-string "request=$COVER"
refused "a multipart cover with a ref_audio part and no source" "$HINT_MULTIPART" \
    --form-string "request=$COVER" -F "ref_audio=@$REF;type=audio/wav"

# No audio at all: a part that would be refused were it decoded.
printf 'not audio' >"$work/junk.wav"
code="$(curl -sS --max-time "$REQUEST_TIMEOUT_S" -o "$work/body" -w '%{http_code}' \
    --form-string 'request={"task_type":"covers","caption":"warm analog synth","duration":4}' \
    -F "audio=@$work/junk.wav;type=audio/wav" "$URL/synth")" || fail "an unknown task: no answer"
[ "$code" = 400 ] || fail "an unknown task: HTTP $code, want 400: $(cat "$work/body")"
grep -qF 'unknown task_type' "$work/body" || fail "an unknown task: not the batch rule's reason: $(cat "$work/body")"
answers "after an unknown task"
echo "ok: a multipart of an unknown task with a junk audio part -- $(cat "$work/body")"

code="$(curl -sS --max-time "$REQUEST_TIMEOUT_S" -o "$work/body" -w '%{http_code}' \
    --form-string 'request={"task_type":"text2music","caption":"warm analog synth","duration":4}' \
    -F "audio=@$work/junk.wav;type=audio/wav" "$URL/synth")" || fail "a text2music with a source: no answer"
[ "$code" = 200 ] || fail "a text2music with a source: HTTP $code, want 200: $(cat "$work/body")"
id="$(sed -n 's/^{"id":"\([^"]*\)"}$/\1/p' "$work/body")"
[ -n "$id" ] || fail "a text2music with a source: no job id in $(cat "$work/body")"
curl -sS --max-time "$REQUEST_TIMEOUT_S" -o /dev/null -X POST "$URL/job?id=$id&cancel=1" ||
    fail "a text2music with a source: no answer to the cancel"
grep -qF "[Server] WARNING: job $id: task 'text2music' ignores the given source: not read" "$work/server.log" ||
    fail "a text2music with a source: no warning naming job $id"
answers "after a text2music with a source"
echo "ok: a text2music with a junk audio part -- job $id, its source unread and said so"
echo "PASS"
