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

# ── Detect GPU ────────────────────────────────────────────────────────────────
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
    log_warn "  → Run '$(basename "$0") migrate-token' or set \$HF_TOKEN"
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

# ── cmd: run ──────────────────────────────────────────────────────────────────
cmd_run() {
  header "🚀 Run Model Server"

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

  # GPU detection
  detect_gpu
  local llama_image ngl
  if $GPU_AVAILABLE; then
    llama_image="$LLAMA_IMAGE_GPU"
    log_info "  GPU: ${GPU_NAME}"
    read -rp "GPU layers to offload --n-gpu-layers [${DEFAULT_NGL}]: " ngl_input
    ngl="${ngl_input:-$DEFAULT_NGL}"
    validate_int "$ngl" "n-gpu-layers" 0
  else
    llama_image="$LLAMA_IMAGE_CPU"
    ngl=0
    log_warn "nvidia-smi not found, running on CPU (slow!)"
  fi

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

  local docker_cmd=(
    docker run -it --rm
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
    trap '[[ -n "$envfile" ]] && rm -f "$envfile"' EXIT
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

  # Clean up env-file after container exits
  "${docker_cmd[@]}" || true
  [[ -n "$envfile" ]] && rm -f "$envfile"
  trap - EXIT
}

# ── cmd: stop ─────────────────────────────────────────────────────────────────
cmd_stop() {
  header "🛑 Stop Model Server"
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    docker stop "$CONTAINER_NAME"
    log_ok "Stopped container: ${CONTAINER_NAME}"
  else
    log_warn "No running container named '${CONTAINER_NAME}' found."
  fi
}

# ── cmd: status ───────────────────────────────────────────────────────────────
cmd_status() {
  header "📊 Server Status"
  if docker inspect "$CONTAINER_NAME" &>/dev/null 2>&1; then
    local state
    state=$(docker inspect -f '{{.State.Status}}' "$CONTAINER_NAME")
    log_ok "Container '${CONTAINER_NAME}' is ${state}"
    docker inspect -f \
      '  Image:   {{.Config.Image}}{{"\n"}}  Started: {{.State.StartedAt}}' \
      "$CONTAINER_NAME"
  else
    log_warn "Container '${CONTAINER_NAME}' is not running."
  fi
}

# ── Usage ─────────────────────────────────────────────────────────────────────
usage() {
  echo -e "
${BOLD}${CYAN}llama-docker-manager${NC} v${VERSION}  llama.cpp model manager

${BOLD}Usage:${NC}
  $(basename "$0") <command> [args]

${BOLD}Commands:${NC}
  ${GREEN}list${NC}               Show installed models
  ${GREEN}download${NC} [repo…]   Download from models.txt, or pass repo IDs directly
  ${GREEN}delete${NC}             Interactively delete a model
  ${GREEN}run${NC}                Launch llama.cpp server (interactive)
  ${GREEN}stop${NC}               Stop the running server container
  ${GREEN}status${NC}             Show server container status

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
  if [[ $# -eq 0 ]]; then usage; exit 0; fi

  load_env_file
  require_cmd docker

  local cmd="$1"; shift

  case "$cmd" in
    list|ls)             cmd_list ;;
    download|dl)         cmd_download "$@" ;;
    delete|rm)           cmd_delete ;;
    run|start)           cmd_run ;;
    stop)                cmd_stop ;;
    status)              cmd_status ;;
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
