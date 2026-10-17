<div align="center">

# 🧠 ANE Load

**A small macOS app with one slider that keeps the Apple Neural Engine busy from 0 to 100 %.**

![macOS 14+](https://img.shields.io/badge/macOS-14%2B-000000?logo=apple&logoColor=white)
![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-M1%E2%80%93M5-555555?logo=apple&logoColor=white)
![Swift](https://img.shields.io/badge/Swift-SwiftUI-F05138?logo=swift&logoColor=white)
![Core ML](https://img.shields.io/badge/Core%20ML-Neural%20Engine-34AADC)
![Version](https://img.shields.io/badge/version-1.0-blue)

</div>

Built for testing power draw, temperatures, fan curves, and how other apps behave while the
Neural Engine (ANE) is busy. One Swift file and one build script, no Xcode project.

## ✨ Features

- **One slider**: target load from 0 to 100 % in steps of 1, plus 0 / 25 / 50 / 75 / 100 % presets.
- **Really on the Neural Engine**: runs a bundled Core ML model with
  `computeUnits = .cpuAndNeuralEngine`, so the GPU is never used. On macOS 14.4+ the window shows
  how many of the model's ops Core ML placed on the ANE.
- **Live numbers**: inferences per second and approximate fp16 TFLOPS.
- **Offline only**: no network code. Check it yourself:
  `grep -nE "URLSession|Network\.|socket|http" Sources/ANELoadApp.swift` finds nothing.
- **Build it yourself**: Xcode Command Line Tools only. There is no prebuilt app in this repo.
- **Any Apple Silicon Mac**: M1 through M5 and newer.

## 📦 Prerequisites

| Need | For |
|------|-----|
| Apple Silicon Mac, macOS 14 or newer | Running the app (14.4+ for the placement check) |
| Xcode Command Line Tools (`xcode-select --install`) | Building (`swiftc`, `codesign`) |
| Python 3, `coremltools`, `numpy` | Regenerating the model only (optional) |

## 🚀 Quick start

```bash
git clone https://github.com/usrnmhcs5he/MacosX-ANE-loader.git
cd MacosX-ANE-loader
./build.sh
open "build/ANE Load.app"
```

What happens:

```
build.sh: swiftc → app bundle + model → ad-hoc codesign
first start: compile model → load onto the ANE → ready → move the slider
```

- You can drag `build/ANE Load.app` into `/Applications`. Gatekeeper doesn't block it, because an app
  you built on your own Mac isn't quarantined.
- The first start takes a few seconds while Core ML compiles the model.

### The window

| Field | Meaning |
|-------|---------|
| **Throughput** | Inferences per second, smoothed |
| **Compute** | Approximate fp16 TFLOPS (throughput × 58 GFLOP per inference) |
| **Placement** | "N of M ops on the Neural Engine". Ideally N equals M |
| **Status** | Model compile and load progress, or the error |

Closing the window quits the app and stops the load.

## ⚙️ How it works

- **The model** (`Model/AneStress.mlpackage`, about 1.2 MB): 12 stacked fp16 3×3 convolutions,
  with 256 channels at 64×64, about **58 GFLOP per inference**. A plain convolution stack is what
  the ANE runs natively. The weights are random numbers.
- **The slider sets a duty cycle.** In every 100 ms window, two worker threads run inferences back
  to back for *slider %* of the window, then sleep. The windows are aligned to the clock, so both
  workers pause at the same time. At 100 % they never sleep.

> [!NOTE]
> macOS has no public API that sets or reports Neural Engine utilization. The percentage is a
> **share of time**, not a hardware counter. Power draw follows it roughly, but not exactly linearly.

## 📁 Files

```
MacosX-ANE-loader/
├── Sources/ANELoadApp.swift     the whole app: window, load engine, placement check
├── Model/AneStress.mlpackage    Core ML model (random weights)
├── make_model.py                regenerates the model with coremltools
├── build.sh                     builds and signs build/ANE Load.app
└── Info.plist                   app bundle metadata
```

`build/` is created by `build.sh` and ignored by git.

## 💡 Usage notes

- **Check that it runs on the ANE**: "ANE Power" should rise and fall with the slider.
  `asitop` and `mactop` show the same counter.
  ```bash
  sudo powermetrics --samplers cpu_power -i 1000 | grep -i ane
  ```
- **Placement below M of M**: if N is lower than M, Core ML moved part of the model to the CPU. The ANE load is then lower than the slider says.
- **Low settings are coarse**: one inference takes a few milliseconds, so 1–3 % isn't precise.
- **Not a benchmark**: the TFLOPS figure counts the convolutions only. Don't compare it across
  chips or macOS versions.
- **Regenerate the model** (optional; `pip` is the only step that uses the network):
  ```bash
  python3 -m venv .venv && .venv/bin/pip install coremltools numpy
  .venv/bin/python make_model.py
  ```
  If you change the model's size, update `gflopPerInference` and the input shape in
  `Sources/ANELoadApp.swift`.

## 🕘 Changelog

| Version | Changes |
|---------|---------|
| **1.0** | First release: slider and presets, duty-cycle load on two workers, throughput and TFLOPS readout, ANE placement check (macOS 14.4+), `build.sh` with ad-hoc signing, `make_model.py` |
