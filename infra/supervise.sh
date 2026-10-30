#!/usr/bin/env bash
# Keep a Spot run alive from your laptop: after each preemption, restart the VM and relaunch the
# (resumable) run in tmux. Exits once the VM stops for any other reason, e.g. run_method.sh
# powering it off when it finishes. Your laptop must stay awake while this runs.
#
#   RUN="PRESS=snapkv RATIO=0.5 bash run_method.sh" nohup bash infra/supervise.sh > supervise.log 2>&1 &
#   VM=kv-a100 ZONE=us-central1-a RUN="..." bash infra/supervise.sh
VM="${VM:-kv-a100}"
ZONE="${ZONE:-us-central1-a}"
: "${RUN:?set RUN to the command to relaunch, as in the examples at the top of this file}"
g() { gcloud compute "$@" --zone "$ZONE" --quiet; }
log() { echo "$(date -u +%H:%M) $*"; }

while true; do
  s=$(g instances describe "$VM" --format="value(status)" 2>&1)
  case "$s" in
    RUNNING|STOPPING|PROVISIONING|STAGING|REPAIRING) sleep 120 ;;
    TERMINATED|STOPPED|SUSPENDED)
      op=$(gcloud compute operations list --filter="targetLink~$VM" --sort-by=~insertTime --limit=1 \
        --format="value(operationType)")
      if [[ "$op" != "compute.instances.preempted" ]]; then
        log "VM stopped (last operation: $op): run finished or failed; check the run log in results/"
        exit 0
      fi
      log "preempted; restarting"
      until g instances start "$VM" >/dev/null 2>&1; do log "start failed (Spot stockout?); retrying in 5 min"; sleep 300; done
      # Wait for SSH, but give up if the VM is preempted again before it comes up; the next pass restarts it.
      until g ssh "$VM" --command true >/dev/null 2>&1; do
        [[ "$(g instances describe "$VM" --format="value(status)" 2>/dev/null)" == RUNNING ]] || break
        sleep 15
      done
      g ssh "$VM" --command "cd KVCompress-survey && tmux new -d -s run '$RUN'" && log "relaunched" ;;
    *) log "unexpected status: $s"; sleep 120 ;;
  esac
done
