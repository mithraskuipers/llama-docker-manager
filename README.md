# llama-docker-manager

Download and run GGUF models locally using llama.cpp. Everything runs inside Docker, so no local GPU drivers or Python needed.


## What you need

- [Docker](https://docs.docker.com/get-docker/)
- An NVIDIA GPU is optional but makes a big difference in speed
- A [HuggingFace token](https://huggingface.co/settings/tokens) if you want to download gated models


## Getting started

```bash
git clone https://github.com/your-username/llama-docker-manager.git
cd llama-docker-manager

# Set up your HF token (see section below)
cp .env.example .env
nano .env

# NVIDIA GPU only run once to install the NVIDIA Container Toolkit
bash setup-nvidia-docker.sh

# Download a model
bash llama-docker-manager.sh download

# Start the server
bash llama-docker-manager.sh run
# Open http://localhost:8080
```


## HuggingFace token

Many models require you to accept terms on HuggingFace before downloading. For those you need a token.

Get one at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough), then put it in your `.env`:

```bash
cp .env.example .env
```

```dotenv
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The `.env` file is gitignored so it won't accidentally get pushed. The token is passed to Docker via `--env-file` rather than `-e KEY=VALUE`, so it stays out of `docker inspect` and process listings.

If you'd rather not use a file, just export it in your shell the script will pick it up:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```


## Commands

| Command | What it does |
|---|---|
| `list` | Show downloaded models |
| `download [repo]` | Download from `models.txt`, or pass a repo ID directly |
| `run` | Pick a model and start the server |
| `stop` | Stop the server |
| `status` | Check if the server is running |
| `delete` | Remove a downloaded model |

```bash
bash llama-docker-manager.sh download
bash llama-docker-manager.sh download org/ModelName:Q4_K_M
bash llama-docker-manager.sh run
bash llama-docker-manager.sh stop
```


## models.txt

List the models you want one HuggingFace repo ID per line. Append `:QUANT` to grab a specific quantization:

```
# models.txt
HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M
mistralai/Mistral-7B-Instruct-v0.3:Q5_K_M
```

Running `download` without arguments gives you an interactive menu of everything in this file.


## Where models end up

Models are saved to `~/models/` on your host outside Docker, outside this repo. They stick around between runs and won't be re-downloaded unless you ask.

```
~/models/
└── HauhauCS__Qwen3.5-9B-Uncensored-HauhauCS-Aggressive--Q4_K_M/
    └── model.gguf
```

Want them somewhere else? Set `MODELS_DIR`:

```bash
MODELS_DIR=/data/models bash llama-docker-manager.sh run
```


## Config

Everything has a sensible default but can be overridden via environment variables:

| Variable | Default | |
|---|---|---|
| `MODELS_DIR` | `~/models` | Where models are stored |
| `MODELS_TXT` | `./models.txt` | Model list |
| `ENV_FILE` | `./.env` | Path to env file |
| `CONTAINER_NAME` | `llama-docker-server` | Docker container name |
| `DEFAULT_CTX` | `8192` | Context window size |
| `DEFAULT_PORT` | `8080` | Host port |
| `LOG_LEVEL` | `info` | `debug` / `info` / `warn` / `error` |


## Files

```
llama-docker-manager/
├── llama-docker-manager.sh   # the script
├── setup-nvidia-docker.sh    # one-time GPU setup
├── models.txt                # your model list
├── .env.example              # token template safe to commit
├── .env                      # your real token gitignored, don't commit
├── .gitignore
└── README.md
```
