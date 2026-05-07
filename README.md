# 🧠 FPGA-Based Neuromorphic SNN Processor
### *Design and Synthesis of a Low-Power Spiking Neural Network Inference Engine*

> **83.76% MNIST accuracy · 0.35 ms latency · 161 mW · 56 µJ/inference**  
> Built entirely in Verilog HDL on a Xilinx Artix-7 FPGA (Basys3) — no IP cores, no shortcuts.

---

## 📸 Demo

<!-- Replace with your actual image path after uploading to repo -->
![FPGA recognizing digit 9](docs/demo.jpg)

*Screen shows digit "9" from MNIST dataset. Basys3 7-segment display shows the live FPGA prediction — in 0.35 ms.*

---

## 📌 Overview

This project implements a complete **Spiking Neural Network (SNN) inference accelerator** on a Xilinx Artix-7 FPGA. The system classifies handwritten MNIST digits in real time:

- A **28×28 pixel image** is sent from a host PC over UART
- The **FPGA runs inference** through a two-layer SNN
- The **predicted digit** is returned over UART and displayed on the 7-segment display and LEDs

SNNs differ from standard neural networks by mimicking biological neurons — they only "fire" (spike) when their membrane potential crosses a threshold, making them inherently **more energy-efficient** than conventional AI architectures.

---

## ⚡ Key Results

| Metric | Value |
|---|---|
| MNIST Test Accuracy | **83.76%** (8,376 / 10,000 images) |
| Inference Latency | **0.35 ms** |
| Clock Frequency | **83.3 MHz** |
| Total On-Chip Power | **161 mW** |
| Energy per Inference | **56 µJ** |
| LUT Utilisation | **29%** (5,961 / 20,800) |
| Flip-Flops | **36%** (14,778 / 41,600) |
| BRAM Tiles | **33 / 50** (66%) |
| DSP Blocks | **0** (pure LUT arithmetic) |

---

## 🏗️ Architecture

```
Host PC (Python Client)
        │
        │  UART 115200 baud  (0xAB + 784 bytes)
        ▼
┌─────────────────────────────────────────────┐
│              Basys3 Artix-7 FPGA             │
│                                             │
│  uart_rx ──► pixel_buffer ──► snn_inference │
│                                      │      │
│                              uart_tx ◄──────│
│                                             │
│  LED[3:0]  → predicted digit (binary)       │
│  LED[11:8] → winner spike count             │
│  LED[15]   → result ready indicator         │
│  7-seg     → digit display                  │
└─────────────────────────────────────────────┘
        │
        │  UART response  (0xBB + 1 byte)
        ▼
Host PC (displays prediction)
```

### SNN Network Topology
```
Input Layer    Hidden Layer    Output Layer
784 neurons →  128 neurons  →  10 neurons
     └──── 8 temporal timesteps per inference ────┘
```

Each neuron implements the **Leaky Integrate-and-Fire (LIF)** model:
- Membrane leakage via arithmetic right-shift (`DECAY_SHIFT`)
- Synaptic current integration
- Threshold comparison and spike emission
- Voltage reset on spike
- All in **32-bit signed fixed-point arithmetic**

---

## 📁 Repository Structure

```
fpga-snn-basys3/
│
├── rtl/                          # Verilog RTL source files
│   ├── snn_inference_live.v      # SNN engine — 431-line LIF model, 13-state FSM
│   ├── uart_rx.v                 # 8N1 UART receiver (16× oversampling)
│   ├── uart_tx.v                 # 8N1 UART transmitter
│   ├── pixel_buffer.v            # BRAM-backed 784-byte dual-port pixel RAM
│   └── top_basys3_snn_live.v     # Basys3 top-level (LEDs, 7-seg, UART pipeline)
│
├── constraints/
│   └── basys3_snn_live.xdc       # Vivado XDC pin constraints for Basys3
│
├── sim/
│   └── tb_snn_live.v             # Self-checking testbench with CSV logging
│
├── host/
│   └── ide_snn_client.py         # Python host client (send images, eval accuracy)
│
│
├── docs/
│   └── demo.jpg                  # Demo photo (replace with your image)
│
└── README.md
```

---

## 🔧 Engineering Challenges Solved

### 1. BRAM Inference Failure (98% → 29% LUT utilisation)
Vivado was inferring the W1 weight matrix (784×128, ~100 KB) as LUT-RAM instead of BRAM36, causing **98% LUT fill** and **TNS = −3815 ns**.

**Root cause:** A combinational read `wire w1v = W1[w1_addr]` prevented BRAM inference — BRAM requires a registered (clocked) read port.

**Fix:** Converted to synchronous registered read:
```verilog
always @(posedge clk)
    w1v <= W1[w1_addr];
```
LUT usage dropped from **98% → 29%** in a single synthesis pass. ✅

---

### 2. FSM Deadlock After First Inference
The `S_DONE` state was missing the `state <= S_IDLE` transition, causing the FSM to lock after the first inference.

**Fix:** Added a single state transition line. Validated across 10 consecutive back-to-back inferences with zero inter-frame reset.

---

### 3. LUT-RAM Mux Tree Elimination
Vivado built 128:1 mux trees for dynamically-addressed register arrays, wasting ~2,500 LUTs.

**Fix:** Added `(* ram_style = "distributed" *)` attributes and removed async reset for-loops (incompatible with LUT-RAM).

---

### 4. UART TX Result Drop
The 1-cycle `valid` pulse from the SNN engine could be missed if the UART transmitter was busy.

**Fix:** Added a `send_pending` handshake register and a 3-state result sender FSM (`RS_IDLE → RS_MARKER → RS_RESULT`) with `tx_busy` edge detection.

---

## 🚀 Getting Started

### Prerequisites
- Vivado 2021 (or later)
- Basys3 FPGA board
- Python 3.8+

### Python Host Client Setup
```bash
pip install pyserial tensorflow pillow numpy matplotlib
```

### UART Protocol
```
PC  → FPGA : 0xAB (sync byte) + 784 pixel bytes (row-major, uint8, 115200 baud)
FPGA → PC  : 0xBB (marker byte) + 1 byte prediction (0x00–0x09)
```

### FPGA Programming
1. Open Vivado → Create new project → Add all files from `rtl/` and `constraints/`
2. Set `top_basys3_snn_live.v` as top module
3. Run Synthesis → Implementation → Generate Bitstream
4. Program the Basys3 board
5. Run `ide_snn_client.py` on your PC

---

## 📊 Synthesis Report Summary

| Resource | Used | Available | Utilisation |
|---|---|---|---|
| LUTs | 5,961 | 20,800 | 29% |
| Flip-Flops | 14,778 | 41,600 | 36% |
| BRAM Tiles | 33 | 50 | 66% |
| IOBs | 32 | 106 | 30% |
| DSPs | 0 | 90 | 0% |

**Timing:** WNS = −0.404 ns @ 83.3 MHz constraint | True Fmax ≈ 97.1 MHz  
**Power:** 0.161 W total (0.089 W dynamic + 0.072 W static)

---

## 🖥️ Host Application — `ide_snn_client.py`

The Python host application runs on your PC and communicates with the FPGA over UART. It includes a full **Tkinter-based GUI** for drawing digits and a command-line interactive menu.

### Features
- 🎨 **Draw mode** — paint window to draw a digit with your mouse, sent live to the FPGA
- 🔢 **MNIST test mode** — send any of the 10,000 MNIST test images by index
- 📁 **File mode** — send any `.png` / `.jpg` image directly
- 📊 **Accuracy eval** — run the full 10k MNIST evaluation and report accuracy
- 🗂️ **Generate mode** — create `input.mem` for Vivado simulation (no FPGA needed)

### Setup
```bash
pip install pyserial tensorflow pillow numpy matplotlib
```

### Usage
```bash
# Interactive menu (recommended)
python ide_snn_client.py

# Send a specific MNIST image by index
python ide_snn_client.py --port COM3 --index 7

# Draw a digit using the paint window
python ide_snn_client.py --port COM3 --draw

# Run full 10k accuracy evaluation
python ide_snn_client.py --port COM3 --eval

# Generate input.mem for Vivado simulation (no FPGA needed)
python ide_snn_client.py --generate --index 42
```

---

## 👨‍💻 Team

| Name | Role |
|---|---|
| **Aditya Das** | RTL Design, BRAM Optimization, UART Pipeline, Host Client |
| **Xwin J Thomas** | SNN Architecture, LIF Neuron Model, Verification |
| **Sarbeswar Dash** | PyTorch Training, Weight Quantisation, Testbench |
| **A. Architha** | Desktop App, Evaluation Dashboard, Integration |

*M.Tech — ECE Department | NIELIT*

---

## 📄 License

This project is open-source under the [MIT License](LICENSE).

---

## 🔗 References

- [Maass, W. (1997). Networks of spiking neurons: The third generation of neural models](https://doi.org/10.1016/S0893-6080(97)00011-7)
- [Xilinx UG901 — Vivado Design Suite: Synthesis](https://docs.xilinx.com/r/en-US/ug901-vivado-synthesis)
- [Digilent Basys3 Reference Manual](https://digilent.com/reference/programmable-logic/basys-3/reference-manual)
- [LeCun et al. — MNIST Handwritten Digit Database](http://yann.lecun.com/exdb/mnist/)
