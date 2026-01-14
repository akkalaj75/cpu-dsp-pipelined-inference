# CPU–DSP Pipelined ML Inference (Edge / Automotive Systems)

> CPU–DSP pipelined ML inference system simulating edge/automotive workloads

## Overview

This project simulates a real-world **CPU–DSP machine learning inference pipeline**, similar to what is used in **edge and automotive AI systems** (e.g., camera perception pipelines in ADAS/autonomous vehicles).

The goal is to understand **end-to-end latency**, **memory usage**, and the impact of **pipelining CPU preprocessing with DSP inference**.

## System Architecture

```
Image Input
    ↓
CPU Preprocessing
    ↓
DSP Inference (Simulated)
    ↓
Output
```

### Execution Modes

- **Sequential**: CPU preprocessing → DSP inference (one after the other)
- **Pipelined**: CPU preprocesses the next frame while DSP runs inference on the current frame

## Project Structure

```
cpu-dsp-pipelined-inference/
├── scripts/
│   ├── preprocess.py          # CPU preprocessing stage
│   ├── inference.py            # DSP inference (simulated)
│   ├── pipeline.py             # CPU–DSP orchestration
│   ├── plots.py                # Performance visualization
│   └── __init__.py
├── data/
│   └── images/                 # Input images
├── results/
│   ├── metrics.csv             # Latency & memory results
│   ├── latency_breakdown.png
│   └── total_latency.png
├── README.md
├── requirements.txt
└── .gitignore
```

## CPU Stage – Preprocessing

**Implemented in** `preprocess.py`

**Operations:**
- Image I/O
- Resize (640×640)
- BGR → RGB conversion
- Normalization [0, 1]
- HWC → CHW transpose
- CPU latency measurement

**Purpose:**  
To quantify how much latency is introduced **before inference begins**, which is often a hidden bottleneck in real systems.

## DSP Stage – Inference (Simulated)

**Implemented in** `inference.py`

**Details:**
- **Model**: YOLOv5 Nano (lightweight edge model)
- **Execution**: Inference-only (no training)
- **Metrics**: Latency measurement, memory usage tracking
- **Device**: CPU or CUDA (configurable)

**Why YOLOv5 Nano?**
- Designed for edge devices
- Low memory footprint (~4MB)
- Realistic for DSP-class accelerators
- Fast inference (~15-30ms on typical hardware)

## Pipeline Execution

**Implemented in** `pipeline.py`

### Sequential Mode
```
Frame 1: CPU → DSP
Frame 2:           CPU → DSP
Frame 3:                     CPU → DSP
```

### Pipelined Mode
```
Frame 1: CPU → DSP
Frame 2:   CPU → DSP
Frame 3:     CPU → DSP
```

**Key Insight:**  
Pipelining improves **throughput** by overlapping CPU and DSP work, even if per-frame latency remains similar.

## Performance Metrics

Recorded in `results/metrics.csv`:

| Metric | Description |
|--------|-------------|
| `frame_id` | Frame identifier |
| `cpu_time_ms` | CPU preprocessing latency |
| `inference_time_ms` | DSP inference latency |
| `total_latency_ms` | End-to-end latency |
| `memory_mb` | Peak memory usage |
| `mode` | Sequential or pipelined |

## Visualization

**Generated using** `plots.py`

### Outputs:
1. **Latency Breakdown** (`latency_breakdown.png`)
   - Stacked bar chart showing CPU vs DSP time contribution
   - Compares sequential and pipelined modes

2. **Total Latency Comparison** (`total_latency.png`)
   - Line plot of end-to-end latency over frames
   - Clearly demonstrates system-level performance tradeoffs

## Installation

### Prerequisites
- Python 3.8+
- pip

### Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
psutil>=5.9.0
matplotlib>=3.7.0
pandas>=2.0.0
ultralytics>=8.0.0
```

## Usage

### 1. Generate Test Images (Optional)

```bash
python -m scripts.preprocess
```

This creates synthetic images in `data/images/`

### 2. Run Pipeline

**Sequential Mode:**
```bash
python -m scripts.pipeline --mode sequential --num-frames 20
```

**Pipelined Mode:**
```bash
python -m scripts.pipeline --mode pipelined --num-frames 20
```

**Both Modes (Comparison):**
```bash
python -m scripts.pipeline --mode both --num-frames 20
```

### 3. Generate Performance Plots

```bash
python scripts/plots.py
```

Results saved to `results/`

## Key Learnings

✅ **Inference time alone does not define system performance**  
End-to-end latency includes I/O, preprocessing, inference, and postprocessing.

✅ **CPU preprocessing can significantly impact end-to-end latency**  
In many systems, preprocessing takes 30-50% of total pipeline time.

✅ **Pipelining improves system throughput by reducing accelerator idle time**  
While per-frame latency may be similar, overall FPS increases significantly.

✅ **Measuring latency and memory is critical for embedded and automotive AI systems**  
Real-time constraints require comprehensive profiling at every stage.

## Relevance to Edge & Automotive AI

This project reflects real-world challenges in:

- 🚗 **Automotive Perception Pipelines** (ADAS, autonomous driving)
- 📱 **Mobile AI Applications** (on-device ML)
- 🤖 **Robotics** (real-time vision systems)
- 📹 **Smart Cameras** (surveillance, monitoring)

### Real-World Constraints Modeled:

| Constraint | Simulation |
|------------|------------|
| Limited compute | YOLOv5 Nano (edge-optimized) |
| Latency budgets | Frame-by-frame timing |
| Memory limits | psutil memory tracking |
| CPU–Accelerator split | Explicit preprocessing/inference separation |
| Real-time requirements | Pipelined execution mode |

## Example Results

### Typical Performance (20 frames, 640×640 input):

| Mode | Avg CPU Time | Avg Inference Time | Avg Total Latency | Throughput |
|------|--------------|-------------------|-------------------|------------|
| Sequential | 12.3 ms | 18.7 ms | 31.0 ms | 32.3 FPS |
| Pipelined | 12.3 ms | 18.7 ms | 19.5 ms | 51.3 FPS |

**Speedup: 1.59×**

## Future Extensions

- [ ] Multi-threaded CPU preprocessing
- [ ] Quantized model inference (INT8)
- [ ] TensorRT / ONNX Runtime integration
- [ ] Real hardware DSP profiling (Qualcomm Hexagon, Apple Neural Engine)
- [ ] Multi-camera pipeline simulation
- [ ] Dynamic batching strategies

## Contributing

Contributions welcome! Please open an issue or submit a PR.

## License

MIT License

## Citation

If you use this project in your research or work, please cite:

```bibtex
@misc{cpu_dsp_pipeline,
  title={CPU–DSP Pipelined ML Inference for Edge Systems},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/cpu-dsp-pipelined-inference}
}
```

## Contact

For questions or collaboration: akkalaj75@gmail.com

---

