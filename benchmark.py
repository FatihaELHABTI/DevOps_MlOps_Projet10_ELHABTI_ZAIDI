import argparse
import time
import json
import os
import psutil
import platform
from pathlib import Path
import numpy as np
from tflite_runtime.interpreter import Interpreter

def run_benchmark(device: str = "CPU", precision: str = "fp32"):
    model_map = {
        "fp32": "models/model_float32.tflite",
        "int8": "models/model_int8.tflite"
    }

    model_path = model_map.get(precision)
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Check DVC pull.")

    print(f"Loading real {precision.upper()} model: {model_path} on {device}")
    arch = platform.machine()
    print(f"Detected architecture: {arch}")

    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get input shape and dtype from the model
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    
    print(f"Model expects input shape: {input_shape}, dtype: {input_dtype}")

    # Create dummy input with the correct dtype
    if input_dtype == np.uint8:
        # For INT8 quantized models - generate random UINT8 data
        dummy_input = np.random.randint(0, 256, size=input_shape, dtype=np.uint8)
    elif input_dtype == np.int8:
        # For some INT8 models that use signed integers
        dummy_input = np.random.randint(-128, 128, size=input_shape, dtype=np.int8)
    else:
        # For FP32 and other float models
        dummy_input = np.random.random(input_shape).astype(input_dtype)

    # Warmup
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()

    num_frames = 100
    total_time = 0.0
    process = psutil.Process(os.getpid())
    peak_memory_mb = process.memory_info().rss / 1024 / 1024

    for _ in range(num_frames):
        start = time.time()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        end = time.time()
        total_time += (end - start)

        mem = process.memory_info().rss / 1024 / 1024
        if mem > peak_memory_mb:
            peak_memory_mb = mem

    avg_latency_ms = (total_time / num_frames) * 1000
    fps = num_frames / total_time
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)

    metrics = {
        "device": device,
        "precision": precision,
        "architecture": arch,
        "input_dtype": str(input_dtype),
        "avg_latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 2),
        "model_size_mb": round(model_size_mb, 1),
        "peak_memory_mb": round(peak_memory_mb, 1),
        "total_frames": num_frames,
        "notes": "Real TFLite inference"
    }

    print(f"Benchmark complete: {fps:.1f} FPS, {avg_latency_ms:.2f} ms latency")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge AI Benchmark")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--device", type=str, default="CPU")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "int8"])

    args = parser.parse_args()

    results = run_benchmark(device=args.device, precision=args.precision)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")