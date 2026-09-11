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
  "gpu_model": "H100",
  "gpu_count": 8,
  "gpus": null,
  "blob_root": null,
  "blob_prefix": "lucayu/sglang",
  "local_root": null,
  "tunnel": false,
  "tunnel_name": null,
  "idle_seconds": 30,
  "poll_seconds": 1,
  "max_utilization": 5,
  "max_memory_mib": 256,
  "matrix_size": 4096,
  "worker_stop_seconds": 10,
  "worker_retry_seconds": 60,
  "retry_seconds": 10,
  "relay": {
    "enabled": true,
    "host": "62.146.171.45",
    "user": "jumper",
    "port": 22,
    "identity_file": "/tmp/singularity-secrets/id_jumper",
    "known_hosts_file": "/tmp/singularity-secrets/known_hosts",
    "forwards": [
      {
        "remote_socket": "/home/jumper/.sglang-relay/api.sock",
        "local_port": 30000
      }
    ],
    "connect_timeout": 15,
    "server_alive_interval": 30,
    "server_alive_count_max": 3
  },
  "aml": {
    "input_name": "lucayu",
    "datastore_uri": "azureml://datastores/zhiyuhe/paths/",
    "repository": "https://github.com/1ucayu/singularity_auto_placeholder.git",
    "ref": "3abb4d996aa701c7836b0ad90f1417e81d6b7c8f",
    "keep_alive_seconds": 8553600,
    "clone_retry_seconds": 30
  }
}
PLACEHOLDER_PROFILE_JSON
  until git clone --no-checkout -- https://github.com/1ucayu/singularity_auto_placeholder.git "$repo/repo" &&
    git -C "$repo/repo" fetch origin 3abb4d996aa701c7836b0ad90f1417e81d6b7c8f &&
    git -C "$repo/repo" checkout --detach FETCH_HEAD; do
    rm -rf -- "$repo/repo"
    sleep 30
  done
  exec bash "$repo/repo/scripts/aml_start.sh" --config "$repo/profile.json" --blob-root "$1"
) &
sleep 8553600 &
wait "$!"
' _ "$(cat <<'PLACEHOLDER_AML_MOUNT'
${{inputs.lucayu}}
PLACEHOLDER_AML_MOUNT
)"
