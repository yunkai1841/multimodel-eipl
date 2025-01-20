# multimodel-eipl

Benchmark multiple eipl models with TensorRT.

## dependencies
- TensorRT 8.6.1
- gflags

## build
```bash
cmake -B build -S .
cmake --build build
```

## run

Create TensorRT engine from onnx model
```bash
trtexec --onnx=<path-to-onnx-model> --saveEngine=<path-to-trt-engine>
```

Run multimodel-eipl
```bash
./build/multitask_benchmark --engine <path-to-trt-engine> --total-models <number-of-models> --loops <number-of-loops>
```
