# CPU–DSP Pipelined ML Inference (Edge / Automotive Systems)

CPU–DSP pipelined ML inference system simulating edge/automotive workloads

\## Overview

This project simulates a real-world \*\*CPU–DSP machine learning inference pipeline\*\*, similar to what is used in \*\*edge and automotive AI systems\*\* (e.g., camera perception pipelines).



The goal is to understand \*\*end-to-end latency\*\*, \*\*memory usage\*\*, and the impact of \*\*pipelining CPU preprocessing with DSP inference\*\*.



---



\## System Architecture



Image Input

↓

CPU Preprocessing

↓

DSP Inference (Simulated)

↓

Output



yaml

Copy code



\### Execution Modes

\- \*\*Sequential\*\*: CPU preprocessing → DSP inference (one after the other)

\- \*\*Pipelined\*\*: CPU preprocesses the next frame while DSP runs inference on the current frame



---



\## Project Structure



cpu-dsp-pipelined-inference/

├── scripts/

│ ├── preprocess.py # CPU preprocessing stage

│ ├── inference.py # DSP inference (simulated)

│ ├── pipeline.py # CPU–DSP orchestration

│ ├── plots.py # Performance visualization

│ └── init.py

├── data/

│ └── images/ # Input images

├── results/

│ ├── metrics.csv # Latency \& memory results

│ ├── latency\_breakdown.png

│ └── total\_latency.png

├── README.md

└── .gitignore



yaml

Copy code



---



\## CPU Stage – Preprocessing

Implemented in `preprocess.py`.



Operations:

\- Image I/O

\- Resize

\- Normalization

\- CPU latency measurement



\*\*Purpose:\*\*  

To quantify how much latency is introduced \*\*before inference begins\*\*, which is often a hidden bottleneck in real systems.



---



\## DSP Stage – Inference (Simulated)

Implemented in `inference.py`.



Details:

\- YOLOv5 Nano (lightweight edge model)

\- Inference-only execution

\- Latency measurement

\- Memory usage tracking



\*\*Why YOLOv5 Nano?\*\*

\- Designed for edge devices

\- Low memory footprint

\- Realistic for DSP-class accelerators



---



\## Pipeline Execution

Implemented in `pipeline.py`.



\### Sequential Mode

CPU → DSP → CPU → DSP



shell

Copy code



\### Pipelined Mode

CPU(frame N+1) || DSP(frame N)



yaml

Copy code



\*\*Key Insight:\*\*  

Pipelining improves \*\*throughput\*\* by overlapping CPU and DSP work, even if per-frame latency remains similar.



---



\## Performance Metrics



Recorded in `results/metrics.csv`:

\- `cpu\_time\_ms`

\- `inference\_time\_ms`

\- `total\_latency\_ms`

\- `memory\_mb`

\- `mode` (sequential / pipelined)



---



\## Visualization



Generated using `plots.py`:

\- \*\*Latency breakdown\*\* (CPU vs DSP)

\- \*\*Total latency comparison\*\* (Sequential vs Pipelined)



These plots clearly demonstrate system-level performance tradeoffs.



---



\## How to Run



\### Install dependencies

```bash

pip install torch opencv-python psutil matplotlib pandas ultralytics

Run pipeline

bash

Copy code

python -m scripts.pipeline

Generate plots

bash

Copy code

python scripts/plots.py

Key Learnings

Inference time alone does not define system performance



CPU preprocessing can significantly impact end-to-end latency



Pipelining improves system throughput by reducing accelerator idle time



Measuring latency and memory is critical for embedded and automotive AI systems



Relevance to Edge \& Automotive AI

This project reflects real-world challenges in:



CPU–DSP coordination



Latency budgeting



Memory-constrained environments



Real-time perception pipelines



Author

Jyothin





