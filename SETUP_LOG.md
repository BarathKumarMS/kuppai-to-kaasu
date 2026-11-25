# Kuppai-to-Kaasu: Setup Log and Troubleshooting Notes

Running record of the build-environment setup (WSL2 Ubuntu 22.04 on Windows). Last updated: 23 Nov 2026.

## Current status

| Item | Status |
|---|---|
| WSL memory/swap config | Done (5.7 GB RAM, 10 GB swap) |
| Python 3.12 venv at `~/venv` | Done |
| ExecuTorch 1.6.0 (prebuilt wheel) | Done, `import executorch` works |
| Repo `~/executorch` on tag `v1.6.0` + submodules | Done |
| Llama export dependencies (`install_requirements.sh`) | In progress |
| Hugging Face licence for Llama 3.2 | Accepted |
| Hugging Face Read token | To confirm |
| Llama 3.2 1B download | Not started |
| Export to `.pte` | Not started |
| Phone push, app build, classifier, demo | Not started |

## What happened, in order

### 1. First install attempt hung
- `./install_executorch.sh` sat on "Building wheel for executorch" with an idle CPU.
- WSL had only the default 2.8 GB RAM and 1 GB swap, so `.wslconfig` had not taken effect.
- Fix: created `C:\Users\<you>\.wslconfig` (`memory=6GB`, `swap=10GB`), ran `wsl --shutdown`, confirmed with `free -h`.

### 2. Python version rejected
- Error: `ExecuTorch does not support python version 3.10.12: must satisfy ">=3.11,<3.16"`.
- Ubuntu 22.04 ships Python 3.10. The project README's "Python 3.10+" is out of date for ExecuTorch main.
- `uv venv --python 3.12` kept restarting its download (slow link).
- The deadsnakes PPA returned `503 Service Unavailable` for `libpython3.12-dev`.
- Fix: retried apt with `--fix-missing -o Acquire::Retries=5`, then recreated the venv with `python3.12 -m venv ~/venv`.

### 3. Source build failed on the Qualcomm SDK
- The install script auto-enabled the Qualcomm (QNN) and OpenVINO backends. CMake then ran `download_qnn_sdk.py`, a 2.8 GB download.
- It failed with `IncompleteRead` after 5 retries. The unneeded SDK was the cause, not the machine.
- `install_executorch.sh --help` has no flag to disable backends. Setting `CMAKE_ARGS=-DEXECUTORCH_BUILD_QNN=OFF ...` was ignored (the QNN download started again).
- Side lesson: `pip-out` stays tiny and no compiler runs during the QNN download phase, so file counts and `top` looked idle even though data was flowing.

### 4. Workaround: prebuilt wheel
- `pip install executorch` installed **executorch 1.6.0**.
- `executorch.__version__` does not exist (harmless `AttributeError`). Use `pip show executorch`.
- Checked out the matching tag so the export scripts match the wheel:
  ```bash
  cd ~/executorch
  git checkout v1.6.0
  git submodule update --init --recursive
  ```

### 5. Llama export dependencies (in progress)
- Running `bash examples/models/llama/install_requirements.sh` (clones torchtune at a pinned commit, installs lm_eval, tiktoken, tokenizers, and so on).
- Watch for any line that reinstalls `torch` (`Uninstalling torch-...`). If it appears, stop and check, because a changed torch can break the wheel.

## Commands that work (keep these)

```bash
# Every new Ubuntu window
source ~/venv/bin/activate

# Network speed check
a=$(awk '/eth0/{print $2}' /proc/net/dev); sleep 15; b=$(awk '/eth0/{print $2}' /proc/net/dev); echo "$(( (b-a)/15/1024 )) KB/s"

# Version check
pip show executorch
```

## Next steps

1. Finish `install_requirements.sh`, then verify:
   ```bash
   cd ~/executorch
   python -c "import torch, torchao; print(torch.__version__, torchao.__version__)"
   python -c "import examples.models.llama.export_llama_lib; print('export ok')"
   ```
2. Hugging Face login and download (`hf auth login`, then `hf download meta-llama/Llama-3.2-1B-Instruct --include "original/*" --local-dir ~/.llama/checkpoints/Llama3.2-1B-Instruct`), then `export CK=$HOME/.llama/checkpoints/Llama3.2-1B-Instruct/original`.
3. Export to `.pte` (Step 6 from the guide). If it says `Killed`, close other apps and rerun.
4. Copy to Windows, `adb push`, build the LlamaDemo app, then the classifier and extra features.
5. Record real measurements for the main README results table.

## Time estimates (at about 1.5 MB/s)

| Step | Estimate |
|---|---|
| Export dependencies | 3-8 min |
| Llama 3.2 1B download (~2.5 GB) | 25-35 min |
| `.pte` export | 10-25 min |
| Copy and `adb push` | 1-3 min |

## Corrections to the main README and guide

- Python requirement: 3.11 to 3.15 for current ExecuTorch, not "3.10+".
- Setup step 1 should mention checking out a release tag that matches the installed `executorch` wheel.
- Source build of ExecuTorch is optional. The prebuilt wheel is enough for exporting Llama.
