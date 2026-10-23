#!/usr/bin/env bash
# Keep a Spot run alive from your laptop: after each preemption, restart the VM and relaunch the
# (resumable) run in tmux. Exits once the VM stops for any other reason, e.g. run_baseline.sh
# powering it off when it finishes. Your laptop must stay awake while this runs.
#
#   nohup bash infra/supervise.sh > supervise.log 2>&1 &
#   VM=kv-a100 ZONE=us-central1-a RUN="bash run_baseline.sh" bash infra/supervise.sh
VM="${VM:-kv-a100}"
ZONE="${ZONE:-us-central1-a}"
RUN="${RUN:-bash run_baseline.sh}"
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
        log "VM stopped (last operation: $op): run finished or failed; check results/run_baseline.log"
        exit 0
      fi
      log "preempted; restarting"
      until g instances start "$VM" >/dev/null 2>&1; do log "start failed (Spot stockout?); retrying in 5 min"; sleep 300; done
      until g ssh "$VM" --command true >/dev/null 2>&1; do sleep 15; done
      g ssh "$VM" --command "cd KVCompress-survey && tmux new -d -s run '$RUN'" && log "relaunched" ;;
    *) log "unexpected status: $s"; sleep 120 ;;
  esac
done
