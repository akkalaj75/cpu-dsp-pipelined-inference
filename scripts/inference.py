import torch
import time
import psutil
import os

class InferenceEngine:
    """
    Simulates DSP-side ML inference.
    Measures inference latency and memory usage.
    """

    def __init__(self):
        # Load lightweight model (edge-friendly)
        self.model = torch.hub.load(
            'ultralytics/yolov5',
            'yolov5n',
            pretrained=True
        )
        self.model.eval()

    def run_inference(self, input_image):
        """
        Runs inference and measures latency and memory usage.
        """

        process = psutil.Process(os.getpid())

        # Memory before inference
        mem_before = process.memory_info().rss / (1024 * 1024)

        start_time = time.time()

        # Run inference (DSP-simulated)
        _ = self.model(input_image)

        end_time = time.time()

        # Memory after inference
        mem_after = process.memory_info().rss / (1024 * 1024)

        inference_time_ms = (end_time - start_time) * 1000
        memory_used_mb = mem_after - mem_before

        return inference_time_ms, memory_used_mb
