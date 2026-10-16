# throttle-gpu: NVIDIA GPU Temperature Monitor and Thermal Guard

`throttle-gpu` is a Python NVIDIA GPU temperature monitor and thermal protection tool for CUDA training and other GPU-heavy workloads. It watches GPU temperatures while code runs and stops work when a GPU gets too hot. Use it as a Python API or from the command line.

- **`ThermalGuard`**: a `with` block that stops when a GPU overheats.
- **`run_isolated`**: runs a function in a child process and kills it on overheat. This is the hard stop.
- **`throttle-gpu run -- CMD`**: wraps any command, such as a training script, and kills its whole process tree on overheat.
- **`throttle-gpu watch`**: logs temperatures until a limit trips.

Requires an NVIDIA GPU with a working driver. It reads temperatures through NVML (`nvidia-ml-py`, optional) and falls back to `nvidia-smi`.

## Why use this on an MSI Raider GE78?

The MSI Raider GE78 HX is available in configurations with NVIDIA GeForce RTX Laptop GPUs and is designed for high performance. Long CUDA training, rendering, or other sustained GPU work can keep the GPU under load for extended periods. `throttle-gpu` gives those jobs a configurable temperature limit and can stop a command or isolated Python function when that limit is exceeded. This is useful when running long work on a laptop such as the [Raider GE78 HX](https://www.msi.com/Laptop/Raider-GE78-HX-14VX/Overview). Configurations vary. The tool monitors the GPU sensor and stops work; it does not control fans or replace the laptop's built-in thermal protections.

## Install

```bash
pip install "git+https://github.com/QuantumNovice/throttle-gpu.git"
```

To enable the NVML temperature backend, install the `nvml` extra directly from GitHub:

```bash
pip install "throttle-gpu[nvml] @ git+https://github.com/QuantumNovice/throttle-gpu.git"
```

Without the extra, `throttle-gpu` uses the `nvidia-smi` command supplied by the NVIDIA driver.

## Usage

### 1. Context manager

```python
from throttle_gpu import ThermalGuard, ThermalLimitExceeded

try:
    with ThermalGuard(max_temp_c=83, consecutive=3, warn_temp_c=78,
                      on_warning=lambda t, lim: print(f"GPU at {t:.0f} °C")):
        train_model()
except ThermalLimitExceeded as exc:
    print(f"Stopped: {exc}")
```

For long loops, call `guard.check()` at a known point:

```python
with ThermalGuard(max_temp_c=83) as guard:
    for batch in loader:
        guard.check()          # raises ThermalLimitExceeded if tripped
        step(batch)
```

### 2. Hard stop in a child process

Thread interrupts can't reach native code, such as a long CUDA call. `run_isolated` runs the function in a separate process and kills it instead:

```python
from throttle_gpu import run_isolated

def train(epochs):          # must be a module-level function
    ...
    return metrics

metrics = run_isolated(train, args=(10,), max_temp_c=83)
```

### 3. Command line

```bash
throttle-gpu status                       # current temperature of each GPU
throttle-gpu watch --max-temp 83          # log until a limit trips (exit code 3)
throttle-gpu run --max-temp 83 -- python train.py
```

`run` stops the command and everything it started. Exit codes: `0` success (or the command's own code), `2` setup or monitoring error, `3` thermal limit reached.

## Guard rails

| Setting | Default | Meaning |
|---|---|---|
| `max_temp_c` | `85` | Trip when any GPU is strictly hotter than this. |
| `consecutive` | `2` | Samples in a row above the limit before tripping. Filters out spikes. |
| `interval_s` | `1.0` | Seconds between samples. |
| `warn_temp_c` | `None` | Optional early-warning threshold with a callback. |

The guard fails closed: if a temperature reading fails, the work stops with `GPUMonitorError`. It does not continue unmonitored.

## Limits

- Temperatures come from the GPU sensor. Time to trip depends on `interval_s * consecutive`. Heat can rise faster than that between samples, so keep a margin below the GPU's own throttle point.
- `ThermalGuard` interrupts Python code only between bytecodes. A blocking native call is not stopped until it returns. Use `run_isolated` or `throttle-gpu run` for a hard stop.
- `run_isolated` needs a module-level target and picklable arguments and results (the `spawn` start method).
- Only NVIDIA GPUs are supported at the moment.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md). Tests use a fake temperature source, so they run without a GPU.

## Documentation

The [API reference and use case guide](https://quantumnovice.github.io/throttle-gpu/) are published from the `main` branch's `/docs` folder. When source code changes, GitHub Actions uses Python's `pydoc` to regenerate API pages from function and class docstrings and commit the HTML to that folder. The home page includes installation steps and workload examples. Select **Deploy from a branch**, branch `main`, folder `/docs` in the repository's GitHub Pages settings. To generate the API HTML locally, run:

```bash
python scripts/generate_api_docs.py
```

## License

MIT. See [LICENSE](LICENSE).
