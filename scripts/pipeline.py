import os
import csv
from .preprocess import preprocess_image
from .inference import InferenceEngine


import time

IMAGE_DIR = "data/images"
RESULTS_FILE = "results/metrics.csv"


def run_sequential_pipeline(image_paths):
    """
    Runs CPU preprocessing followed by DSP inference sequentially.
    """
    engine = InferenceEngine()
    results = []

    for img_path in image_paths:
        image, cpu_time = preprocess_image(img_path)
        inf_time, mem_used = engine.run_inference(image)

        total_latency = cpu_time + inf_time

        results.append({
            "image": os.path.basename(img_path),
            "cpu_time_ms": cpu_time,
            "inference_time_ms": inf_time,
            "total_latency_ms": total_latency,
            "memory_mb": mem_used,
            "mode": "sequential"
        })

    return results


def run_pipelined_pipeline(image_paths):
    """
    Simulates pipelined execution:
    CPU preprocesses next image while DSP runs inference.
    """
    engine = InferenceEngine()
    results = []

    prev_image = None
    prev_cpu_time = None

    for img_path in image_paths:
        image, cpu_time = preprocess_image(img_path)

        if prev_image is not None:
            inf_time, mem_used = engine.run_inference(prev_image)
            total_latency = prev_cpu_time + inf_time

            results.append({
                "image": os.path.basename(img_path),
                "cpu_time_ms": prev_cpu_time,
                "inference_time_ms": inf_time,
                "total_latency_ms": total_latency,
                "memory_mb": mem_used,
                "mode": "pipelined"
            })

        prev_image = image
        prev_cpu_time = cpu_time

    # Run inference for last image
    inf_time, mem_used = engine.run_inference(prev_image)
    total_latency = prev_cpu_time + inf_time

    results.append({
        "image": "last_frame",
        "cpu_time_ms": prev_cpu_time,
        "inference_time_ms": inf_time,
        "total_latency_ms": total_latency,
        "memory_mb": mem_used,
        "mode": "pipelined"
    })

    return results


def save_results(results):
    os.makedirs("results", exist_ok=True)

    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    image_paths = [
        os.path.join(IMAGE_DIR, img)
        for img in os.listdir(IMAGE_DIR)
        if img.endswith(".jpg") or img.endswith(".png")
    ]

    print("Running sequential pipeline...")
    sequential_results = run_sequential_pipeline(image_paths)

    print("Running pipelined pipeline...")
    pipelined_results = run_pipelined_pipeline(image_paths)

    all_results = sequential_results + pipelined_results
    save_results(all_results)

    print("Results saved to results/metrics.csv")
