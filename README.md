# llama-docker-manager

Download and run GGUF models locally using llama.cpp. Everything runs inside Docker, so no local GPU drivers or Python needed.

---

## What you need

- [Docker](https://docs.docker.com/get-docker/)
- An NVIDIA GPU is optional but makes a big difference in speed
- A [HuggingFace token](https://huggingface.co/settings/tokens) if you want to download gated models

---

## Getting started

```bash
git clone https://github.com/your-username/llama-docker-manager.git
cd llama-docker-manager

# Set up your HF token (see section below)
cp .env.example .env
nano .env

# NVIDIA GPU only: run once to install the NVIDIA Container Toolkit
bash setup-nvidia-docker.sh

# Download a model
bash llama-docker-manager.sh download

# Start the server
bash llama-docker-manager.sh run

# Open in browser
http://localhost:8080
```

---

## Use the self-hosted LLM from other applications (API usage)

Once the server is running, your local LLM is not limited to the browser UI. It exposes a network-accessible endpoint that can be used as a backend for other tools, scripts, and applications.

### Base endpoint

`http://host.docker.internal:8080`

This endpoint can be used as a local LLM API host for any application that supports HTTP-based or OpenAI-style LLM backends (depending on the client/tooling you use).

### Example integrations

You can plug this endpoint into:

- VS Code AI extensions
- Local agent frameworks (LangChain, custom agents, automation scripts)
- Python or Node.js applications
- Internal tools replacing cloud LLM APIs
- Chat UIs or dashboards
- CI/CD automation or security tooling pipelines

### Example usage (generic HTTP concept)

Most clients will ask for something like:

- **Base URL:** `http://host.docker.internal:8080`
- **API Host:** same value
- **Inference endpoint:** same value

From there, the client will handle request formatting.

### Why this matters

This turns your setup into a fully local LLM service:

- No external API calls required
- No token usage or billing
- Fully controllable environment
- Works across multiple applications simultaneously

---

## HuggingFace token

Many models require you to accept terms on HuggingFace before downloading. For those, you need a token.

Get one here: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) (read access is enough)

Then add it to your `.env`:

```bash
cp .env.example .env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

The `.env` file is gitignored, so it won't accidentally get pushed. The token is passed to Docker via `--env-file` rather than `-e KEY=VALUE`, so it stays out of `docker inspect` and process listings.

If you'd rather not use a file, you can export it instead:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

---

## Commands


| Command           | Description                                            |
| ----------------- | ------------------------------------------------------ |
| `list`            | Show downloaded models                                 |
| `download [repo]` | Download from `models.txt`, or pass a repo ID directly |
| `run`             | Pick a model and start the server                      |
| `stop`            | Stop the server                                        |
| `status`          | Check if the server is running                         |
| `delete`          | Remove a downloaded model                              |


### Examples

```bash
bash llama-docker-manager.sh download
bash llama-docker-manager.sh download org/ModelName:Q4_K_M
bash llama-docker-manager.sh run
bash llama-docker-manager.sh stop
```

---

## models.txt

List the models you want, one HuggingFace repo ID per line. Append `:QUANT` to select a specific quantization:

```plaintext
# models.txt
HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive:Q4_K_M
mistralai/Mistral-7B-Instruct-v0.3:Q5_K_M
```

Running `download` without arguments gives you an interactive menu of everything in this file.

---

## Where models end up

Models are stored on your host system outside Docker:

```plaintext
~/models/
└── HauhauCS__Qwen3.5-9B-Uncensored-HauhauCS-Aggressive--Q4_K_M/
    └── model.gguf
```

They persist between runs and will not be re-downloaded unless explicitly requested.

To change location:

```bash
MODELS_DIR=/data/models bash llama-docker-manager.sh run
```

---

## Configuration

All settings can be overridden via environment variables:


| Variable         | Default               | Description             |
| ---------------- | --------------------- | ----------------------- |
| `MODELS_DIR`     | `~/models`            | Where models are stored |
| `MODELS_TXT`     | `./models.txt`        | Model list file         |
| `ENV_FILE`       | `./.env`              | Path to env file        |
| `CONTAINER_NAME` | `llama-docker-server` | Docker container name   |
| `DEFAULT_CTX`    | `8192`                | Context window size     |
| `DEFAULT_PORT`   | `8080`                | Host port               |
| `LOG_LEVEL`      | `info`                | Logging level           |


---

## Project structure

```plaintext
llama-docker-manager/
├── llama-docker-manager.sh   # main control script
├── setup-nvidia-docker.sh    # one-time GPU setup
├── models.txt                # model list
├── .env.example              # safe template
├── .env                      # your HF token (gitignored)
├── .gitignore
└── README.md
```

---

## Summary

This project gives you a fully Dockerized local LLM server based on llama.cpp, with:

- Easy model downloading from HuggingFace
- GPU acceleration support (optional)
- Persistent model storage
- A simple server UI at [http://localhost:8080](http://localhost:8080)
- A reusable API endpoint at `http://host.docker.internal:8080` for integration into any tool or application