#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  llama-docker-manager.sh
#  Manages download, storage, and serving of GGUF models via Docker.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
IFS=$'\n\t'

# ── Version ───────────────────────────────────────────────────────────────────
readonly VERSION="1.1.0"

# ── Config (override via environment) ────────────────────────────────────────
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_TXT="${MODELS_TXT:-${SCRIPT_DIR}/models.txt}"
ENV_FILE="${ENV_FILE:-${SCRIPT_DIR}/.env}"
LLAMA_IMAGE_GPU="${LLAMA_IMAGE_GPU:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
LLAMA_IMAGE_CPU="${LLAMA_IMAGE_CPU:-ghcr.io/ggml-org/llama.cpp:server}"
CONTAINER_NAME="${CONTAINER_NAME:-llama-docker-server}"
DEFAULT_CTX="${DEFAULT_CTX:-8192}"
DEFAULT_PORT="${DEFAULT_PORT:-8080}"
DEFAULT_NGL="${DEFAULT_NGL:-999}"
LOG_LEVEL="${LOG_LEVEL:-info}"   # debug | info | warn | error

# ── Colors (disabled automatically when not a terminal) ──────────────────────
if [[ -t 1 ]]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; DIM=''; NC=''
fi

# ── Logging ───────────────────────────────────────────────────────────────────
log_debug() { [[ "$LOG_LEVEL" == "debug" ]] && echo -e "${DIM}[DBG]${NC} $*" >&2 || true; }
log_info()  { echo -e "${BLUE}ℹ${NC}  $*"; }
log_ok()    { echo -e "${GREEN}✔${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}⚠${NC}  $*"; }
log_error() { echo -e "${RED}✘${NC}  $*" >&2; }
header()    { echo -e "\n${BOLD}${CYAN}$*${NC}"; echo -e "${DIM}$(printf '─%.0s' {1..52})${NC}"; }

die() { log_error "$*"; exit 1; }

# ── Dependency check ──────────────────────────────────────────────────────────
require_cmd() {
  command -v "$1" &>/dev/null || die "'$1' is required but not installed."
}

# ── Docker health check ───────────────────────────────────────────────────────
#
#  Verifies not just that the `docker` CLI exists, but that it can actually
#  talk to a running daemon. This covers two setups:
#    - WSL, where the daemon lives in Docker Desktop on Windows and is
#      reached through the WSL integration
#    - bare-metal / native Linux, where dockerd runs locally as a service
#
is_wsl() {
  # /proc/version on WSL mentions "microsoft" (WSL2) or "Microsoft" (WSL1)
  grep -qi "microsoft" /proc/version 2>/dev/null
}

check_docker() {
  log_debug "Checking docker connectivity..."

  if docker info &>/dev/null; then
    log_debug "Docker daemon is reachable."
    return 0
  fi

  log_error "Docker command found, but the Docker daemon is not reachable."
  echo ""

  if is_wsl; then
    log_warn "Detected WSL."
    log_warn "  This script expects Docker Desktop running on Windows with WSL integration enabled."
    log_warn "  → Start Docker Desktop on Windows"
    log_warn "  → Docker Desktop → Settings → Resources → WSL Integration"
    log_warn "     enable it for this distro, then click 'Apply & Restart'"
    log_warn "  → Re-run this script once the whale icon in the tray shows 'running'"
  else
    log_warn "Detected a native Linux environment."
    log_warn "  → Check the service status: sudo systemctl status docker"
    log_warn "  → Start it:                 sudo systemctl start docker"
    log_warn "  → Start on boot:            sudo systemctl enable docker"
    log_warn "  → If you're not in the docker group, you may need: sudo docker ..."
    log_warn "     or add yourself: sudo usermod -aG docker \$USER  (then log out/in)"
  fi

  echo ""
  die "Cannot continue without a working Docker connection."
}

# ── HF Token loading ────────────────────────────────────────────────────────
#
#  Priority order:
#    1. .env file next to this script  (HF_TOKEN=xxx)
#    2. $HF_TOKEN environment variable (CI/CD, Docker Compose, etc.)
#
#  The token is NEVER passed as -e KEY=VALUE to `docker run`.
#  Instead we write a temp env-file and pass --env-file so it
#  doesn't appear in `docker inspect` or process listings.
#
load_env_file() {
  [[ -f "$ENV_FILE" ]] || return 0
  local perms
  perms=$(stat -c "%a" "$ENV_FILE" 2>/dev/null || stat -f "%Lp" "$ENV_FILE" 2>/dev/null)
  if [[ "$perms" != "600" ]]; then
    log_warn ".env permissions are ${perms}, tightening to 600..."
    chmod 600 "$ENV_FILE"
  fi
  # Export only valid KEY=VALUE lines; skip comments and blanks
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  log_debug "Loaded .env from ${ENV_FILE}"
}

get_hf_token() {
  # 1. .env file (loaded at startup by load_env_file)
  # 2. environment variable (already set externally, e.g. CI/CD)
  if [[ -n "${HF_TOKEN:-}" ]]; then
    echo "$HF_TOKEN"
    return
  fi
  echo ""
}

# Write a temp env-file with the token; caller must clean it up.
# Usage: make_token_envfile token -> prints path to env-file
make_token_envfile() {
  local token="$1"
  local tmpfile
  tmpfile=$(mktemp /tmp/llama-hf-env.XXXXXX)
  chmod 600 "$tmpfile"
  printf 'HF_TOKEN=%s\n' "$token" > "$tmpfile"
  echo "$tmpfile"
}

# ── Quant / repo helpers ──────────────────────────────────────────────────────
strip_quant()   { echo "${1%%:*}"; }
get_quant()     { local q="${1##*:}"; [[ "$q" == "$1" ]] && echo "" || echo "$q"; }
model_dir_name(){ local c="${1//\//__}"; echo "${c//:/--}"; }
model_display() { local d="${1//__//}"; echo "${d//--/:}"; }

find_gguf() {
  find "$1" -name "*.gguf" -type f 2>/dev/null | sort
}

# ── List installed models ─────────────────────────────────────────────────────
list_installed() {
  mkdir -p "$MODELS_DIR"
  mapfile -t dirs < <(find "$MODELS_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort)
  local result=()
  for d in "${dirs[@]+"${dirs[@]}"}"; do
    result+=("$(basename "$d")")
  done
  printf '%s\n' "${result[@]+"${result[@]}"}"
}

# ── Validate numeric input ────────────────────────────────────────────────────
validate_int() {
  local val="$1" label="$2" min="$3" max="${4:-}"
  if ! [[ "$val" =~ ^[0-9]+$ ]]; then
    die "${label} must be a positive integer, got: '${val}'"
  fi
  if (( val < min )); then
    die "${label} must be >= ${min}, got: ${val}"
  fi
  if [[ -n "$max" ]] && (( val > max )); then
    die "${label} must be <= ${max}, got: ${val}"
  fi
}

# ── GPU detection & verification ─────────────────────────────────────────────
#
#  Three separate checks, because each answers a different question:
#    1. detect_gpu               -> does the HOST have an NVIDIA driver at all?
#    2. check_docker_gpu_runtime -> can DOCKER actually pass a GPU into a
#                                   container? (this is what setup-nvidia-docker.sh
#                                   configures — having a driver on the host is
#                                   not enough on its own)
#    3. verify_gpu_active        -> once a model is running, is it ACTUALLY
#                                   using the GPU, not silently falling back to CPU?
#

detect_gpu() {
  local nsmi
  for candidate in nvidia-smi /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi; do
    if command -v "$candidate" &>/dev/null || [[ -x "$candidate" ]]; then
      nsmi="$candidate"
      GPU_NAME=$("$nsmi" --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown GPU")
      GPU_AVAILABLE=true
      return
    fi
  done
  GPU_AVAILABLE=false
  GPU_NAME=""
}

# Tests whether `docker run --gpus all` actually works end-to-end.
# Sets DOCKER_GPU_OK=true/false and DOCKER_GPU_ERROR on failure.
check_docker_gpu_runtime() {
  DOCKER_GPU_OK=false
  DOCKER_GPU_ERROR=""

  local test_image="alpine:3.19"
  if ! docker image inspect "$test_image" &>/dev/null 2>&1; then
    log_debug "Pulling small test image for GPU passthrough check..."
    docker pull -q "$test_image" &>/dev/null || true
  fi

  local out
  if out=$(docker run --rm --gpus all "$test_image" true 2>&1); then
    DOCKER_GPU_OK=true
  else
    DOCKER_GPU_OK=false
    DOCKER_GPU_ERROR="$out"
  fi
}

# Prints a full GPU diagnostics report: host driver + docker passthrough.
print_gpu_diagnostics() {
  header "🎮 GPU Diagnostics"

  detect_gpu
  if $GPU_AVAILABLE; then
    log_ok "Host driver: NVIDIA GPU detected — ${GPU_NAME}"
  else
    log_warn "Host driver: no NVIDIA GPU/driver detected (nvidia-smi not found)."
    log_warn "  This is expected on CPU-only machines. The server will run on CPU."
    return
  fi

  log_info "Checking whether Docker can access the GPU (this may take a few seconds)..."
  check_docker_gpu_runtime

  if $DOCKER_GPU_OK; then
    log_ok "Docker passthrough: working — containers can use --gpus all."
  else
    log_error "Docker passthrough: NOT working."
    log_warn "  Docker sees a GPU driver on the host, but can't hand a GPU to containers."
    log_warn "  → Run: bash setup-nvidia-docker.sh"
    if is_wsl; then
      log_warn "  → Also make sure Docker Desktop's WSL integration is enabled for this distro"
      log_warn "     and that 'GPU support' isn't disabled in Docker Desktop settings."
    fi
    if [[ -n "$DOCKER_GPU_ERROR" ]]; then
      log_debug "Docker error was: ${DOCKER_GPU_ERROR}"
    fi
  fi
}

# Polls until the running container is confirmed to actually be using the GPU,
# by cross-checking the container's host PID against nvidia-smi's list of
# processes with GPU memory allocated. This is the only way to know for sure —
# a GPU being *visible* to a container doesn't guarantee llama.cpp actually
# offloaded any layers to it (e.g. wrong image, 0 layers requested, OOM fallback).
verify_gpu_active() {
  local max_wait="${1:-20}"
  local nsmi=""
  for candidate in nvidia-smi /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi; do
    if command -v "$candidate" &>/dev/null || [[ -x "$candidate" ]]; then
      nsmi="$candidate"
      break
    fi
  done
  [[ -n "$nsmi" ]] || return 1

  local cpid
  cpid=$(docker inspect -f '{{.State.Pid}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  [[ -n "$cpid" && "$cpid" != "0" ]] || return 1

  local waited=0
  while (( waited < max_wait )); do
    local candidate_pids gpu_pids
    candidate_pids="$cpid $(pgrep -P "$cpid" 2>/dev/null)"
    gpu_pids=$("$nsmi" --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

    for pid in $candidate_pids; do
      if grep -qw "$pid" <<< "$gpu_pids"; then
        return 0
      fi
    done

    sleep 1
    ((waited++))
  done
  return 1
}

# ── Embedded Python downloader ────────────────────────────────────────────────
write_downloader() {
  local dest="$1"
  cat > "${dest}/hf_download.py" << 'PYEOF'
#!/usr/bin/env python3
"""HuggingFace GGUF downloader with resume support."""
import os, sys, fnmatch, hashlib, time
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    from huggingface_hub import HfApi
except ImportError:
    import subprocess
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q",
         "huggingface_hub>=0.22", "tqdm", "requests"],
        stdout=subprocess.DEVNULL,
    )
    import requests
    from tqdm import tqdm
    from huggingface_hub import HfApi

def human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"

def download_file(url: str, dest: Path, headers: dict, total: int) -> None:
    resume_pos = dest.stat().st_size if dest.exists() else 0

    if resume_pos and resume_pos == total:
        print(f"  ✔  Already complete: {dest.name}", flush=True)
        return

    dl_headers = dict(headers)
    if resume_pos and resume_pos < total:
        dl_headers["Range"] = f"bytes={resume_pos}-"
        print(f"  ↻  Resuming {dest.name} from {human_size(resume_pos)}", flush=True)
    else:
        resume_pos = 0  # server may not support range
        print(f"  ⬇  {dest.name}  ({human_size(total)})", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    mode = "ab" if resume_pos else "wb"

    for attempt in range(1, 4):
        try:
            with requests.get(url, headers=dl_headers, stream=True,
                              allow_redirects=True, timeout=60) as resp:
                resp.raise_for_status()
                with tqdm(
                    total=total, initial=resume_pos,
                    unit="B", unit_scale=True, unit_divisor=1024,
                    ncols=80,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                ) as bar:
                    with open(dest, mode) as fh:
                        for chunk in resp.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fh.write(chunk)
                                bar.update(len(chunk))
            return
        except (requests.RequestException, OSError) as exc:
            if attempt < 3:
                wait = 2 ** attempt
                print(f"\n  ⚠  Error: {exc}. Retrying in {wait}s …", flush=True)
                time.sleep(wait)
                resume_pos = dest.stat().st_size if dest.exists() else 0
                dl_headers["Range"] = f"bytes={resume_pos}-"
                mode = "ab"
            else:
                print(f"\n  ✘  Failed after 3 attempts: {exc}", file=sys.stderr)
                sys.exit(1)

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: hf_download.py <repo_id> <glob_pattern>", file=sys.stderr)
        sys.exit(1)

    repo    = sys.argv[1]
    pattern = sys.argv[2]
    token   = os.environ.get("HF_TOKEN") or None
    outdir  = Path("/models")

    api = HfApi()
    try:
        all_files = list(api.list_repo_files(repo, token=token))
    except Exception as exc:
        print(f"\n✘  Could not list repo '{repo}': {exc}", file=sys.stderr)
        sys.exit(1)

    matches = sorted(f for f in all_files if fnmatch.fnmatch(f, pattern))

    if not matches:
        print(f"\n✘  No files matching '{pattern}' in {repo}", file=sys.stderr)
        print("    Available files:", file=sys.stderr)
        for f in all_files:
            print(f"      {f}", file=sys.stderr)
        sys.exit(1)

    print(f"\nFiles to download ({len(matches)}):")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    for filename in matches:
        url  = f"https://huggingface.co/{repo}/resolve/main/{filename}"
        dest = outdir / filename

        r = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        download_file(url, dest, headers, total)

    print("\n✔  All downloads complete!")

if __name__ == "__main__":
    main()
PYEOF
}

# ── cmd: list ─────────────────────────────────────────────────────────────────
cmd_list() {
  header "📦 Installed Models"
  mapfile -t installed < <(list_installed)

  if [[ ${#installed[@]} -eq 0 ]]; then
    log_warn "No models installed yet. Use '$(basename "$0") download'."
    return
  fi

  local i=1
  for dir_name in "${installed[@]}"; do
    local display dir_path gguf_files size
    display=$(model_display "$dir_name")
    dir_path="$MODELS_DIR/$dir_name"
    mapfile -t gguf_files < <(find_gguf "$dir_path")
    size=$(du -sh "$dir_path" 2>/dev/null | cut -f1)

    echo -e "  ${BOLD}${i}.${NC} ${GREEN}${display}${NC}  ${DIM}(${size}, ${#gguf_files[@]} .gguf)${NC}"
    for f in "${gguf_files[@]}"; do
      local fsize
      fsize=$(du -sh "$f" 2>/dev/null | cut -f1)
      echo -e "     ${DIM}└─ $(basename "$f")  (${fsize})${NC}"
    done
    ((i++))
  done
  echo ""
}

# ── download_model ────────────────────────────────────────────────────────────
download_model() {
  local repo_id="$1"
  local hf_token="$2"
  local dir_name dest
  dir_name=$(model_dir_name "$repo_id")
  dest="$MODELS_DIR/$dir_name"

  # Lockfile: prevent concurrent downloads to the same dir
  local lockfile="${dest}.lock"
  if [[ -f "$lockfile" ]]; then
    log_warn "Another download seems to be in progress for ${repo_id} (lockfile: ${lockfile})."
    log_warn "If that's not the case, delete the lockfile and retry."
    return 1
  fi

  if [[ -d "$dest" ]] && [[ -n "$(find_gguf "$dest")" ]]; then
    log_warn "Model already downloaded: ${repo_id}"
    read -rp "  Re-download? [y/N] " confirm
    [[ "${confirm,,}" == "y" ]] || return 0
  fi

  local base_repo quant include_pattern
  base_repo=$(strip_quant "$repo_id")
  quant=$(get_quant "$repo_id")
  include_pattern=$([[ -n "$quant" ]] && echo "*${quant}.gguf" || echo "*.gguf")
  [[ -n "$quant" ]] && log_info "Quantization filter: ${quant}"

  mkdir -p "$dest"
  touch "$lockfile"
  # Ensure lockfile is removed on exit (success or failure)
  trap 'rm -f "${lockfile}"' EXIT

  log_info "Downloading: ${BOLD}${repo_id}${NC}"
  log_info "Destination: ${dest}"
  echo ""

  # Write the downloader into a temp dir, not the model dir
  local tmpdir
  tmpdir=$(mktemp -d /tmp/llama-downloader.XXXXXX)
  trap 'rm -rf "${tmpdir}"; rm -f "${lockfile}"' EXIT

  write_downloader "$tmpdir"

  local docker_args=(
    run --rm -it
    -v "${dest}:/models"
    -v "${tmpdir}/hf_download.py:/hf_download.py:ro"
  )

  # Pass token via env-file (never -e KEY=VALUE) to avoid exposure in `docker inspect`
  local envfile=""
  if [[ -n "$hf_token" ]]; then
    envfile=$(make_token_envfile "$hf_token")
    docker_args+=(--env-file "$envfile")
  fi

  docker_args+=(
    python:3.11-slim
    python3 /hf_download.py "${base_repo}" "${include_pattern}"
  )

  local exit_code=0
  if docker "${docker_args[@]}"; then
    log_ok "Downloaded: ${repo_id}"
    local gguf_count
    gguf_count=$(find_gguf "$dest" | wc -l | tr -d ' ')
    log_info "  ${gguf_count} .gguf file(s) in ${dest}"
  else
    exit_code=1
    log_error "Download failed for: ${repo_id}"
  fi

  [[ -n "$envfile" ]] && rm -f "$envfile"
  rm -rf "$tmpdir"
  rm -f "$lockfile"
  trap - EXIT

  return $exit_code
}

# ── cmd: download ─────────────────────────────────────────────────────────────
cmd_download() {
  header "⬇  Download Models"

  local hf_token
  hf_token=$(get_hf_token)
  if [[ -n "$hf_token" ]]; then
    log_ok "HF token loaded (passed via env-file, not exposed to docker inspect)"
  else
    log_warn "No HF token found. Private/gated models will fail."
    log_warn "  → Copy .env.example to .env and set HF_TOKEN, or export \$HF_TOKEN"
  fi

  # Direct invocation: llama-docker-manager download <repo1> [repo2 …]
  if [[ $# -gt 0 ]]; then
    for repo in "$@"; do
      download_model "$repo" "$hf_token"
    done
    return
  fi

  [[ -f "$MODELS_TXT" ]] || die "models.txt not found at: ${MODELS_TXT}"

  mapfile -t all_models < <(grep -v '^\s*#' "$MODELS_TXT" | grep -v '^\s*$')
  [[ ${#all_models[@]} -gt 0 ]] || { log_warn "models.txt is empty."; return; }

  echo -e "${BOLD}Available models in models.txt:${NC}"
  local i=1
  for m in "${all_models[@]}"; do
    local dir_name dest status=""
    dir_name=$(model_dir_name "$m")
    dest="$MODELS_DIR/$dir_name"
    [[ -n "$(find_gguf "$dest" 2>/dev/null)" ]] && status=" ${GREEN}[installed]${NC}"
    echo -e "  ${BOLD}${i}.${NC} ${m}${status}"
    ((i++))
  done

  echo ""
  echo -e "  ${BOLD}a.${NC} Download ALL"
  echo -e "  ${BOLD}q.${NC} Cancel"
  echo ""
  read -rp "Select model(s) [1-${#all_models[@]}, comma-separated, a, q]: " choice

  case "${choice,,}" in
    q) log_info "Cancelled."; return ;;
    a)
      for m in "${all_models[@]}"; do
        download_model "$m" "$hf_token" && echo ""
      done
      ;;
    *)
      IFS=',' read -ra picks <<< "$choice"
      for pick in "${picks[@]}"; do
        pick="${pick//[[:space:]]/}"
        if [[ "$pick" =~ ^[0-9]+$ ]] && (( pick >= 1 && pick <= ${#all_models[@]} )); then
          download_model "${all_models[$((pick-1))]}" "$hf_token" && echo ""
        else
          log_warn "Invalid selection: '${pick}'"
        fi
      done
      ;;
  esac
}

# ── cmd: delete ───────────────────────────────────────────────────────────────
cmd_delete() {
  header "🗑  Delete Model"

  mapfile -t installed < <(list_installed)
  [[ ${#installed[@]} -gt 0 ]] || { log_warn "No models installed."; return; }

  echo -e "${BOLD}Installed models:${NC}\n"
  local i=1
  for dir_name in "${installed[@]}"; do
    local size
    size=$(du -sh "$MODELS_DIR/$dir_name" 2>/dev/null | cut -f1)
    echo -e "  ${BOLD}${i}.${NC} $(model_display "$dir_name")  ${DIM}(${size})${NC}"
    ((i++))
  done
  echo -e "  ${BOLD}q.${NC} Cancel\n"
  read -rp "Select model to delete [1-${#installed[@]}/q]: " choice

  [[ "${choice,,}" == "q" ]] && { log_info "Cancelled."; return; }

  if [[ "$choice" =~ ^[0-9]+$ ]] && (( choice >= 1 && choice <= ${#installed[@]} )); then
    local dir_name="${installed[$((choice-1))]}"
    local dir_path="$MODELS_DIR/$dir_name"
    local size
    size=$(du -sh "$dir_path" 2>/dev/null | cut -f1)

    echo ""
    echo -e "${RED}${BOLD}About to delete:${NC} $(model_display "$dir_name")  ${DIM}(${size})${NC}"
    echo -e "${DIM}  Path: ${dir_path}${NC}"
    read -rp "  Type 'yes' to confirm: " confirm

    if [[ "$confirm" == "yes" ]]; then
      rm -rf "$dir_path"
      log_ok "Deleted: $(model_display "$dir_name")"
    else
      log_info "Cancelled."
    fi
  else
    log_warn "Invalid selection: '${choice}'"
  fi
}

# ── API usage banner ──────────────────────────────────────────────────────────
#
#  Printed right before the container starts (docker run -it blocks in the
#  foreground and floods the terminal with model logs), so this is the last
#  clean thing the user sees and it stays in their scrollback.
#
print_api_usage_banner() {
  local port="$1"
  local rule
  rule=$(printf '─%.0s' {1..64})

  echo -e "${BOLD}${GREEN}${rule}${NC}"
  echo -e "${BOLD}${GREEN}  ✔ Server starting — here's how to use it${NC}"
  echo -e "${BOLD}${GREEN}${rule}${NC}"
  echo ""
  echo -e "  ${BOLD}Chat in your browser${NC}"
  echo -e "    ${CYAN}http://localhost:${port}${NC}"
  echo ""
  echo -e "  ${BOLD}Use it as an API${NC}  ${DIM}(OpenAI-compatible, no key needed)${NC}"
  echo -e "    Base URL:  ${CYAN}http://localhost:${port}/v1${NC}"
  echo ""
  echo -e "  ${BOLD}Quick test${NC}"
  echo -e "    ${DIM}curl http://localhost:${port}/v1/chat/completions \\\\${NC}"
  echo -e "    ${DIM}  -H \"Content-Type: application/json\" \\\\${NC}"
  echo -e "    ${DIM}  -d '{\"model\":\"local\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'${NC}"
  echo ""
  echo -e "  ${BOLD}Plug into other tools${NC} ${DIM}(VS Code extensions, LangChain, scripts, etc.)${NC}"
  echo -e "    Base URL : ${CYAN}http://localhost:${port}/v1${NC}   ${DIM}(if calling from this machine)${NC}"
  echo -e "    API key  : ${DIM}anything, e.g. \"none\" — it isn't checked${NC}"
  echo -e "    ${DIM}Only use http://host.docker.internal:${port}/v1 if the caller is itself${NC}"
  echo -e "    ${DIM}running inside another Docker container.${NC}"
  echo ""
  echo -e "  ${BOLD}Manage it${NC}"
  echo -e "    Check status : ${DIM}bash $(basename "$0") status${NC}  ${DIM}(or menu option)${NC}"
  echo -e "    Stop server  : ${DIM}bash $(basename "$0") stop${NC}    ${DIM}(or menu option)${NC}"
  echo -e "${BOLD}${GREEN}${rule}${NC}"
  echo ""
}

# ── cmd: run ──────────────────────────────────────────────────────────────────
cmd_run() {
  header "🚀 Create & Start Container"

  local hf_token
  hf_token=$(get_hf_token)

  mapfile -t installed < <(list_installed)
  [[ ${#installed[@]} -gt 0 ]] || { log_warn "No models installed. Run 'download' first."; return; }

  echo -e "${BOLD}Installed models:${NC}\n"
  local i=1
  for dir_name in "${installed[@]}"; do
    echo -e "  ${BOLD}${i}.${NC} $(model_display "$dir_name")"
    ((i++))
  done
  echo ""
  read -rp "Select model [1-${#installed[@]}]: " choice

  if ! [[ "$choice" =~ ^[0-9]+$ ]] || (( choice < 1 || choice > ${#installed[@]} )); then
    log_warn "Invalid selection."; return
  fi

  local dir_name="${installed[$((choice-1))]}"
  local display dir_path
  display=$(model_display "$dir_name")
  dir_path="$MODELS_DIR/$dir_name"

  mapfile -t gguf_files < <(find_gguf "$dir_path")
  [[ ${#gguf_files[@]} -gt 0 ]] || die "No .gguf files found in ${dir_path}"

  local gguf_file
  if [[ ${#gguf_files[@]} -eq 1 ]]; then
    gguf_file="${gguf_files[0]}"
    log_info "Using: $(basename "$gguf_file")"
  else
    echo -e "\n${BOLD}Multiple .gguf files:${NC}"
    local j=1
    for f in "${gguf_files[@]}"; do
      local fsize; fsize=$(du -sh "$f" 2>/dev/null | cut -f1)
      echo -e "  ${BOLD}${j}.${NC} $(basename "$f")  ${DIM}(${fsize})${NC}"
      ((j++))
    done
    read -rp "Select file [1-${#gguf_files[@]}]: " fchoice
    if ! [[ "$fchoice" =~ ^[0-9]+$ ]] || (( fchoice < 1 || fchoice > ${#gguf_files[@]} )); then
      log_warn "Invalid selection."; return
    fi
    gguf_file="${gguf_files[$((fchoice-1))]}"
  fi

  read -rp "Context size [${DEFAULT_CTX}]: " ctx_input
  local ctx="${ctx_input:-$DEFAULT_CTX}"
  validate_int "$ctx" "Context size" 512 131072

  read -rp "Host port [${DEFAULT_PORT}]: " port_input
  local port="${port_input:-$DEFAULT_PORT}"
  validate_int "$port" "Port" 1 65535

  # GPU detection: host driver first, then whether Docker can actually use it
  detect_gpu
  local llama_image ngl use_gpu=false
  if $GPU_AVAILABLE; then
    log_info "  Host GPU: ${GPU_NAME}"
    log_info "  Checking Docker GPU passthrough..."
    check_docker_gpu_runtime

    if $DOCKER_GPU_OK; then
      use_gpu=true
      log_ok "  Docker can access the GPU."
    else
      log_error "  Docker cannot access the GPU (driver is present, but passthrough isn't working)."
      log_warn "    → Run: bash setup-nvidia-docker.sh"
      if is_wsl; then
        log_warn "    → Also check Docker Desktop → Settings → Resources → WSL Integration,"
        log_warn "       and that GPU support is enabled there."
      fi
      log_warn "  Falling back to CPU for this run (will be slow)."
    fi
  else
    log_warn "  No NVIDIA driver detected on the host — running on CPU (slow!)."
  fi

  if $use_gpu; then
    llama_image="$LLAMA_IMAGE_GPU"
    read -rp "GPU layers to offload --n-gpu-layers [${DEFAULT_NGL}]: " ngl_input
    ngl="${ngl_input:-$DEFAULT_NGL}"
    validate_int "$ngl" "n-gpu-layers" 0
  else
    llama_image="$LLAMA_IMAGE_CPU"
    ngl=0
  fi
  GPU_AVAILABLE=$use_gpu

  local rel_path
  rel_path=$(realpath --relative-to="$dir_path" "$gguf_file")

  echo ""
  log_info "Starting llama.cpp server..."
  log_info "  Model     : ${display}"
  log_info "  File      : $(basename "$gguf_file")"
  log_info "  Context   : ${ctx}"
  log_info "  GPU layers: ${ngl}"
  log_info "  URL       : http://localhost:${port}"
  log_info "  Container : ${CONTAINER_NAME}"
  echo ""

  # Clean up any stale/stuck container left behind by a previous forced kill
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    log_warn "Found a leftover container named '${CONTAINER_NAME}' — removing it first."
    docker rm -f "$CONTAINER_NAME" &>/dev/null || true
  fi

  local docker_cmd=(
    docker run -d --rm
    --name "$CONTAINER_NAME"
    -p "${port}:8080"
    -v "${dir_path}:/models:ro"
  )

  $GPU_AVAILABLE && docker_cmd+=(--gpus all)

  # Pass token via env-file
  local envfile=""
  if [[ -n "$hf_token" ]]; then
    envfile=$(make_token_envfile "$hf_token")
    docker_cmd+=(--env-file "$envfile")
  fi

  docker_cmd+=(
    "$llama_image"
    --model "/models/${rel_path}"
    --ctx-size "$ctx"
    --n-gpu-layers "$ngl"
    --host 0.0.0.0
    --port 8080
  )

  echo -e "${DIM}$ ${docker_cmd[*]}${NC}\n"

  # ── Run detached and return control immediately ─────────────────────────────
  #
  #  llama.cpp's own SIGINT handler can hang mid-cleanup (seen in practice with
  #  CUDA teardown), which used to make a foreground `docker run -it` totally
  #  unkillable from the same terminal. Running detached avoids that problem
  #  entirely: this command starts the server and hands control straight back
  #  to the menu. Use "Stop" to shut it down (always force-kills reliably).
  #
  if ! "${docker_cmd[@]}" &>/dev/null; then
    log_error "Failed to start the container."
    [[ -n "$envfile" ]] && rm -f "$envfile"
    return 1
  fi

  # The env-file was only needed to create the container; safe to remove now.
  [[ -n "$envfile" ]] && rm -f "$envfile"

  log_ok "Server started in the background."

  if $use_gpu; then
    log_info "Verifying the model actually loaded onto the GPU (up to 20s)..."
    if verify_gpu_active 20; then
      log_ok "Confirmed: running on GPU (${GPU_NAME})."
    else
      log_warn "Could not confirm GPU usage — the model may have silently fallen back to CPU"
      log_warn "  (common causes: model file too large for VRAM, or --n-gpu-layers too low)."
      log_warn "  Check manually with: nvidia-smi"
    fi
  fi

  echo ""
  print_api_usage_banner "$port"
}

# ── cmd: stop ─────────────────────────────────────────────────────────────────
cmd_stop() {
  header "🛑 Stop & Remove Container"
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    log_info "Stopping '${CONTAINER_NAME}' (up to 3s grace period)..."
    docker stop --time=3 "$CONTAINER_NAME" &>/dev/null || true
    # Force-remove regardless — covers containers stuck/hung on shutdown
    docker rm -f "$CONTAINER_NAME" &>/dev/null || true
    log_ok "Stopped container: ${CONTAINER_NAME}"
  else
    log_warn "No running container named '${CONTAINER_NAME}' found."
  fi
}

# ── cmd: update-image ─────────────────────────────────────────────────────────
#
#  "Create & start container" already recreates the container fresh every
#  time — that part needs no manual step. What it does NOT do is refresh the
#  underlying Docker image: the llama.cpp GPU/CPU images are mutable tags
#  upstream, so once pulled they're cached locally forever unless explicitly
#  re-pulled. This command pulls fresh images, and clears out any leftover
#  container along the way just in case.
#
cmd_reset() {
  header "⬇️  Update llama.cpp Image"

  log_info "Pulling latest GPU image (${LLAMA_IMAGE_GPU})..."
  if docker pull "$LLAMA_IMAGE_GPU"; then
    log_ok "GPU image up to date."
  else
    log_warn "Failed to pull GPU image (check network/registry access)."
  fi

  log_info "Pulling latest CPU image (${LLAMA_IMAGE_CPU})..."
  if docker pull "$LLAMA_IMAGE_CPU"; then
    log_ok "CPU image up to date."
  else
    log_warn "Failed to pull CPU image (check network/registry access)."
  fi

  echo ""
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    log_info "Removing existing container so the next run picks up the new image..."
    docker stop --time=3 "$CONTAINER_NAME" &>/dev/null || true
    docker rm -f "$CONTAINER_NAME" &>/dev/null || true
    log_ok "Container removed."
  fi

  log_ok "Ready — the next 'Create & start container' will use the updated image."
}

# ── cmd: status ───────────────────────────────────────────────────────────────
cmd_status() {
  header "📊 Container Status"
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    local state port
    state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
    log_ok "Container '${CONTAINER_NAME}' is ${state}"
    docker inspect -f \
      '  Image:   {{.Config.Image}}{{"\n"}}  Started: {{.State.StartedAt}}' \
      "$CONTAINER_NAME"

    if [[ "$state" == "running" ]]; then
      port=$(docker inspect -f '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
      if [[ -n "$port" ]]; then
        echo ""
        log_info "  Web UI  : http://localhost:${port}"
        log_info "  API     : http://localhost:${port}/v1"
      fi

      local requested_gpu
      requested_gpu=$(docker inspect -f '{{len .HostConfig.DeviceRequests}}' "$CONTAINER_NAME" 2>/dev/null || echo "0")
      echo ""
      if [[ "$requested_gpu" -gt 0 ]]; then
        log_info "  Checking GPU usage (up to 5s)..."
        if verify_gpu_active 5; then
          log_ok "  GPU: confirmed in use"
        else
          log_warn "  GPU: requested but not confirmed active — may be running on CPU"
        fi
      else
        log_info "  GPU: not requested for this run (running on CPU)"
      fi
    fi
  else
    log_warn "Container '${CONTAINER_NAME}' is not running."
    log_warn "  → Start it with: bash $(basename "$0") run"
  fi
}

# ── Running server info ────────────────────────────────────────────────────────
#
#  Used to show "what's currently running" at the top of the interactive menu.
#
get_running_server_info() {
  docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1 || return 1

  local state
  state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME" 2>/dev/null || echo "")
  [[ "$state" == "running" ]] || return 1

  local src dir_name display port
  src=$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/models"}}{{.Source}}{{end}}{{end}}' "$CONTAINER_NAME" 2>/dev/null)
  dir_name=$(basename "${src:-unknown}")
  display=$(model_display "$dir_name")
  port=$(docker inspect -f '{{(index (index .NetworkSettings.Ports "8080/tcp") 0).HostPort}}' "$CONTAINER_NAME" 2>/dev/null)

  printf '%s|%s\n' "$display" "$port"
}

print_running_banner() {
  local info display port rule
  info=$(get_running_server_info) || return 0
  [[ -z "$info" ]] && return 0

  display="${info%%|*}"
  port="${info##*|}"
  rule=$(printf '─%.0s' {1..64})

  echo -e "${BOLD}${GREEN}${rule}${NC}"
  echo -e "${BOLD}${GREEN}  ● RUNNING${NC}   ${BOLD}${display}${NC}"
  echo -e "    Web UI : ${CYAN}http://localhost:${port}${NC}"
  echo -e "    API    : ${CYAN}http://localhost:${port}/v1${NC}   ${DIM}(OpenAI-compatible)${NC}"
  echo -e "${BOLD}${GREEN}${rule}${NC}"
  echo ""
}

# ── Interactive main menu ──────────────────────────────────────────────────────
main_menu() {
  while true; do
    echo -e "\n${BOLD}${CYAN}llama-docker-manager${NC} ${DIM}v${VERSION}${NC}"
    echo ""
    print_running_banner

    echo -e "${BOLD}What would you like to do?${NC}"
    echo -e "  ${BOLD}1)${NC} List installed models"
    echo -e "  ${BOLD}2)${NC} Download model(s)"
    echo -e "  ${BOLD}3)${NC} Create & start container (run a model)"
    echo -e "  ${BOLD}4)${NC} Stop & remove container"
    echo -e "  ${BOLD}5)${NC} Container status"
    echo -e "  ${BOLD}6)${NC} Delete a model"
    echo -e "  ${BOLD}7)${NC} GPU diagnostics"
    echo -e "  ${BOLD}8)${NC} Update llama.cpp image (pull latest)"
    echo -e "  ${BOLD}q)${NC} Quit"
    echo ""
    read -rp "Choice: " menu_choice
    echo ""

    case "${menu_choice,,}" in
      1) cmd_list ;;
      2) cmd_download ;;
      3) cmd_run ;;
      4) cmd_stop ;;
      5) cmd_status ;;
      6) cmd_delete ;;
      7) print_gpu_diagnostics ;;
      8) cmd_reset ;;
      q|quit|exit) log_info "Bye!"; exit 0 ;;
      *) log_warn "Invalid choice: '${menu_choice}'" ;;
    esac

    echo ""
    read -rp "Press Enter to continue..." _
  done
}

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  echo -e "
${BOLD}${CYAN}llama-docker-manager${NC} v${VERSION}  llama.cpp model manager

${BOLD}Usage:${NC}
  $(basename "$0")                Launch the interactive menu
  $(basename "$0") <command> [args]

${BOLD}Commands:${NC}
  ${GREEN}menu${NC}               Launch the interactive menu (same as no args)
  ${GREEN}list${NC}               Show installed models
  ${GREEN}download${NC} [repo…]   Download from models.txt, or pass repo IDs directly
  ${GREEN}delete${NC}             Interactively delete a model
  ${GREEN}run${NC}                Create & start a new container in the background, then return
  ${GREEN}stop${NC}               Stop & remove the running container
  ${GREEN}status${NC}             Show server container status
  ${GREEN}gpu${NC}                Run GPU diagnostics (host driver + Docker passthrough)
  ${GREEN}update-image${NC}       Pull the latest llama.cpp image(s) and clear old container

${BOLD}HF Token:${NC}
  Copy ${DIM}.env.example${NC} to ${DIM}.env${NC} and fill in your token:
    ${DIM}HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx${NC}
  The ${DIM}.env${NC} file is gitignored and never committed.
  Alternatively export ${DIM}\$HF_TOKEN${NC} in your shell (CI/CD, Docker Compose).

${BOLD}Environment overrides:${NC}
  ${DIM}MODELS_DIR${NC}       Model storage (default: ~/models)
  ${DIM}MODELS_TXT${NC}       Path to model list (default: ./models.txt)
  ${DIM}ENV_FILE${NC}         Path to .env file (default: ./.env)
  ${DIM}CONTAINER_NAME${NC}   Docker container name (default: llama-docker-server)
  ${DIM}DEFAULT_CTX${NC}      Default context size (default: 8192)
  ${DIM}DEFAULT_PORT${NC}     Default port (default: 8080)
  ${DIM}LOG_LEVEL${NC}        debug | info | warn | error (default: info)

${BOLD}Examples:${NC}
  $(basename "$0") list
  $(basename "$0") download
  $(basename "$0") download org/Model:Q4_K_M
  $(basename "$0") run
  $(basename "$0") stop
"
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
  load_env_file
  require_cmd docker
  check_docker

  if [[ $# -eq 0 ]]; then main_menu; exit 0; fi

  local cmd="$1"; shift

  case "$cmd" in
    menu)                main_menu ;;
    list|ls)             cmd_list ;;
    download|dl)         cmd_download "$@" ;;
    delete|rm)           cmd_delete ;;
    run|start)           cmd_run ;;
    stop)                cmd_stop ;;
    status)              cmd_status ;;
    gpu)                 print_gpu_diagnostics ;;
    update-image|reset)  cmd_reset ;;
    version|--version)   echo "llama-docker-manager v${VERSION}" ;;
    help|--help|-h)      usage ;;
    *)
      log_error "Unknown command: '${cmd}'"
      usage
      exit 1
      ;;
  esac
}

main "$@"
