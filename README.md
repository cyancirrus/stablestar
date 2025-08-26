# Moonlight

**Moonlight** is a C++/CUDA sandbox for experimenting with **GPU-focused numerical computing**.
The goal is to prototype low-level GPU kernels and pipelines, while keeping the workflow strict and ergonomic (sanitizers, LSP, Make targets).

Rust is my main environment for systems + numerical work, but GPU kernels will live here — with `ffi` bridges back into Rust where needed.

---

## Build & Run

### Development

```bash
make              # build debug target
make run          # run debug build
```

### Release

```bash
make release      # optimized build (versioned under ./target/build-<VERSION>/)
```

### Diagnostics / Checks

Moonlight integrates multiple safety nets:

* **Host C++ checks:**

  * `-fsanitize=address,undefined,leak`
* **Device CUDA checks:**

  * `make memcheck` → run under `compute-sanitizer --tool memcheck`
  * `make racecheck` → run under `compute-sanitizer --tool racecheck`

```bash
make memcheck
make racecheck
```

These are analogous to Rust’s strict runtime checks — immediate feedback when you trip on UB.

---

## Example Pipeline

Currently, the project has a simple demo pipeline:

```cpp
vector<float> pipeline(
    int n,
    float c,
    vector<float>& x,
    vector<float>& y
) {
    int blocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
    float *d_x, *d_y;

    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_y, n * sizeof(float));

    cudaMemcpy(d_x, x.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    add<<<blocks, BLOCKSIZE>>>(n, d_x, d_y);
    scale<<<blocks, BLOCKSIZE>>>(n, c, d_y);

    cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);

    return y;
}
```

This kernel-pipeline pattern is the core abstraction:

* **low-level CUDA kernels** (e.g. `add`, `scale`)
* **C++ host pipelines** wrapping them
* future **Rust FFI bindings** will connect at the `vector<float>&` boundary.

---

## Next Steps

* Implement core matrix operations (matmul, transpose, reductions).
* Explore memory layouts (row-major, shared memory tiling).
* Build minimal Rust FFI surface on top.
* Compare performance vs. pure Rust `StellarMath` experiments.

---

## Notes

* Built & tested on **Arch Linux** with `CUDA 13.0`.
* GPU: **NVIDIA RTX 2070** (compute capability `sm_75`).
* Compiler: `nvcc` with `clangd` LSP integration.
