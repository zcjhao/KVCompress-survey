#!/usr/bin/env bash
# Create a single-L4 GPU VM on GCP for the KV-cache experiments.
# Run this from your laptop (needs gcloud, already authenticated).
#
#   bash infra/create_vm.sh          # Spot VM (default): ~40% cheaper, but GCP can preempt it
#   SPOT=0 bash infra/create_vm.sh   # on-demand VM: never preempted
#
# Everyday commands afterwards:
#   gcloud compute ssh $VM_NAME --zone $ZONE
#   gcloud compute instances stop  $VM_NAME --zone $ZONE   # stop GPU billing (disk is still billed)
#   gcloud compute instances start $VM_NAME --zone $ZONE
#   gcloud compute instances delete $VM_NAME --zone $ZONE  # remove completely
set -euo pipefail

VM_NAME="${VM_NAME:-kv-l4}"
ZONE="${ZONE:-us-east1-b}"            # your project has L4 quota in us-east1
MACHINE="${MACHINE:-g2-standard-4}"   # 1x L4 (24 GB), 4 vCPU, 16 GB RAM: the smallest L4 shape
DISK_GB="${DISK_GB:-100}"             # image (50 GB) + env (~10 GB) + Llama-3.1-8B weights (~16 GB) + datasets
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"

EXTRA_ARGS=()
if [[ "${SPOT:-1}" == "1" ]]; then
  # On preemption the VM stops rather than being deleted, so the disk (model + results) survives.
  EXTRA_ARGS+=(--provisioning-model=SPOT --instance-termination-action=STOP)
fi

gcloud compute instances create "$VM_NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE" \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-project=deeplearning-platform-release \
  --image-family="$IMAGE_FAMILY" \
  --boot-disk-size="${DISK_GB}GB" \
  --boot-disk-type=pd-balanced \
  --metadata=install-nvidia-driver=True \
  "${EXTRA_ARGS[@]}"

echo "Created $VM_NAME. SSH in with: gcloud compute ssh $VM_NAME --zone $ZONE"
echo "Remember to STOP it when you are done: gcloud compute instances stop $VM_NAME --zone $ZONE"
