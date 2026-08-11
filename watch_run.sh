#!/bin/bash
# Watch a detached run and return on ANY of: finished, died, stalled.
#
#   watch_run.sh <logfile> <pattern-that-matches-the-process> [poll-seconds]
#
# The plain `until grep -q "^EXIT=" log; do sleep; done` waiter only returns when
# the job writes EXIT=. If the job is killed -- OOM, a SIGHUP from the launching
# shell, a segfaulting worker taking the session down -- that line never appears
# and the wait never ends, which is indistinguishable from "still running" and
# has stalled several turns. This returns in that case too, and says which.
#
# Exit codes: 0 finished, 1 died without EXIT=, 2 log stopped growing.
set -u
log=$1
pattern=$2
poll=${3:-300}
stall_limit=3  # consecutive polls with no new output before calling it stalled

previous=""
stalls=0
while true; do
    if grep -q "^EXIT=" "$log" 2>/dev/null; then
        echo "=== FINISHED after $(( SECONDS / 60 ))m ==="
        tail -25 "$log"
        exit 0
    fi
    if ! pgrep -f "$pattern" > /dev/null 2>&1; then
        echo "=== DIED after $(( SECONDS / 60 ))m: no process matching '$pattern', and the log has no EXIT= ==="
        echo "the job was killed rather than finishing; the log's last lines:"
        tail -25 "$log" 2>/dev/null
        exit 1
    fi
    size=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "$size" = "$previous" ]; then
        stalls=$(( stalls + 1 ))
    else
        stalls=0
    fi
    if [ "$stalls" -ge "$stall_limit" ]; then
        echo "=== STALLED: $log has not grown in $(( stall_limit * poll / 60 )) minutes, process still alive ==="
        tail -25 "$log" 2>/dev/null
        exit 2
    fi
    previous=$size
    sleep "$poll"
done
