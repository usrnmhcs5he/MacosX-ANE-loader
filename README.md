# ANE Load: a slider for the Apple Neural Engine

`ANE Load` is a small macOS app with one slider. It keeps the **Apple Neural Engine (ANE)** busy for
the chosen share of time, from **0 to 100 %**, and shows live throughput.
Use it to test power draw, temperatures, fan curves, or how other apps behave while the ANE is busy.

- **Apple Silicon only:** any M chip, M1 through M5 and newer, on macOS 14 or newer.
- **Offline only:** the app has no network code. It runs a bundled Core ML model and nothing else.
  You can check: `grep -nE "URLSession|Network\.|socket|http" Sources/ANELoadApp.swift` finds nothing.
- **You build it yourself:** one script, Xcode Command Line Tools only (full Xcode isn't needed).
  There is no prebuilt app in this repo.

## 1. Build and run

```bash
xcode-select --install        # once, if the Command Line Tools aren't installed yet
git clone https://github.com/usrnmhcs5he/MacosX-ANE-loader.git
cd MacosX-ANE-loader
./build.sh
open "build/ANE Load.app"
```

- `build.sh` compiles `Sources/ANELoadApp.swift` with `swiftc`, puts the model into the app bundle
  and signs the app ad hoc (`codesign --sign -`).
- You can drag `build/ANE Load.app` into `/Applications`. Gatekeeper doesn't block it, because an app
  you built on your own Mac isn't quarantined.
- The first start takes a few seconds while Core ML compiles the model for the Neural Engine.

## 2. Use it

| Control or field | Meaning |
|---|---|
| **Slider** | Target load, 0–100 % in steps of 1 |
| **0 / 25 / 50 / 75 / 100 %** | Presets |
| **Throughput** | Inferences per second, smoothed |
| **Compute** | Approximate fp16 TFLOPS (throughput × 58 GFLOP per inference) |
| **Placement** | How many of the model's ops Core ML put on the Neural Engine (macOS 14.4 or newer) |
| **Status** | Model compile and load progress, or the error |

Closing the window quits the app and stops the load.

## 3. Check that it really runs on the Neural Engine

```bash
sudo powermetrics --samplers cpu_power -i 1000 | grep -i ane
```

**ANE Power** should rise and fall with the slider. `asitop` and `mactop` show the same counter.
If **Placement** says something like "12 of 25 ops", Core ML has moved part of the model to the CPU.
The load is then lower than the slider shows.

## 4. How the load works

- **The model** (`Model/AneStress.mlpackage`) is a stack of 12 fp16 3×3 convolutions, with
  256 channels at 64×64. That is about **58 GFLOP per inference**. A plain convolution stack is
  what the Neural Engine runs natively. The weights are random.
- **Neural Engine, not GPU:** the app loads the model with `computeUnits = .cpuAndNeuralEngine`, so Core ML
  can't use the GPU.
- **The slider sets a duty cycle.** In every 100 ms window, two worker threads run inferences back to back
  for *slider %* of the window, then sleep. Both workers use the same clock-aligned windows,
  so the ANE is fully idle in the rest of each window. At 100 % the workers never sleep.

macOS has no public API that sets or reports Neural Engine utilization. **The percentage is a share of time,
not a hardware counter.** Power draw follows it roughly, but not exactly linearly.

## 5. Files

| File | What it is |
|---|---|
| `Sources/ANELoadApp.swift` | The whole app: SwiftUI window, load engine, placement check |
| `Model/AneStress.mlpackage` | The Core ML model (about 1.2 MB, random weights) |
| `make_model.py` | Generates the model with `coremltools` |
| `build.sh` | Builds and signs `build/ANE Load.app` |
| `Info.plist` | App bundle metadata |

## 6. Regenerating the model (optional)

The bundled model contains only random numbers. To build it yourself instead:

```bash
python3 -m venv .venv && .venv/bin/pip install coremltools numpy
.venv/bin/python make_model.py
```

`pip` downloads `coremltools` from PyPI; this is the only step that uses the network. If you change the model's size in
`make_model.py`, update `gflopPerInference` and the input shape in `Sources/ANELoadApp.swift`.

## 7. Limits

- **Granularity:** one inference takes a few milliseconds, so very low settings (1–3 %) are coarse.
- **Core ML decides placement.** Apple can change which ops run on the ANE between macOS versions.
  The **Placement** field shows the current split.
- **Not a benchmark.** The TFLOPS figure counts the convolutions only and is approximate. Different
  chips and macOS versions aren't directly comparable with it.
- **Heat and battery:** running at 100 % for a long time drains the battery noticeably. macOS manages
  temperatures and fans on its own.
