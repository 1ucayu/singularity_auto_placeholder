# Generated from configs/example.json; regenerate for your own settings.
env -u BASH_ENV bash --noprofile --norc -c 'set -m
cleanup() {
  trap - EXIT TERM INT HUP
  for pid in $(jobs -pr); do kill -TERM -- "-$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
trap "exit 143" TERM
trap "exit 130" INT
trap "exit 129" HUP
(
  repo="$(mktemp -d /tmp/singularity-placeholder.XXXXXX)" || exit 1
  cat > "$repo/profile.json" <<'"'"'PLACEHOLDER_PROFILE_JSON'"'"'
{
  "schema_version": 1,
  "gpu_model": "A100",
  "gpu_count": 4,
  "gpus": null,
  "blob_root": null,
  "blob_prefix": "",
  "local_root": null,
  "tunnel": true,
  "tunnel_name": null,
  "idle_seconds": 30,
  "poll_seconds": 1,
  "max_utilization": 5,
  "max_memory_mib": 256,
  "matrix_size": 4096,
  "worker_stop_seconds": 10,
  "worker_retry_seconds": 60,
  "retry_seconds": 10,
  "aml": {
    "input_name": "storage",
    "datastore_uri": "azureml://datastores/my_datastore/paths/my-project/",
    "repository": "https://github.com/1ucayu/singularity_auto_placeholder.git",
    "ref": "main",
    "keep_alive_seconds": 86400,
    "clone_retry_seconds": 30
  }
}
PLACEHOLDER_PROFILE_JSON
  until git clone --no-checkout -- https://github.com/1ucayu/singularity_auto_placeholder.git "$repo/repo" &&
    git -C "$repo/repo" fetch origin main &&
    git -C "$repo/repo" checkout --detach FETCH_HEAD; do
    rm -rf -- "$repo/repo"
    sleep 30
  done
  exec bash "$repo/repo/scripts/aml_start.sh" --config "$repo/profile.json" --blob-root "$1"
) &
sleep 86400 &
wait "$!"
' _ "$(cat <<'PLACEHOLDER_AML_MOUNT'
${{inputs.storage}}
PLACEHOLDER_AML_MOUNT
)"
