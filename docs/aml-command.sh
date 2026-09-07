env -u BASH_ENV bash --noprofile --norc -c '
set -m
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
  until git clone --depth 1 https://github.com/1ucayu/singularity_auto_placeholder.git "$repo/repo"; do
    rm -rf -- "$repo/repo"
    sleep 30
  done
  exec bash "$repo/repo/scripts/aml_start.sh" --blob-root "$1" --gpu-count 8 --tunnel --tunnel-name aml-lucayu
) &
sleep 8553600 &
wait "$!"
' _ "${{inputs.lucayu}}"
