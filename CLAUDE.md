# Vulkan Demo Project

## Overview
Vulkan rendering demo, will evolve into SPH fluid simulation engine.

## Optimization Directions

### Async Compute (compute + render overlap)
- Use two queues from queue family 0: one for compute, one for graphics
- Compute and render run in parallel via semaphore synchronization
- Expected improvement: 170fps -> 280-320fps based on current OpenGL profiling

### Double Buffer Decoupling
- Two copies of position buffer: Buffer A (compute writes), Buffer B (render reads)
- Use transfer queue (family 2, dedicated DMA engine) for async A->B copy
- Compute never waits for render, achieving zero rendering overhead on simulation
- Alternative: fully unsynchronized read (acceptable visual artifacts for SPH, zero overhead, no extra memory)

### Precise Pipeline Barriers
- Replace OpenGL's global glMemoryBarrier with per-buffer Vulkan barriers
- Specify exact src/dst stage masks and specific buffers
- SPH stages are strictly serial (0->1->3->4->8->9), no inter-stage parallelism possible
- Benefit comes from narrower barrier scope, not from parallelizing stages

## Current SPH Performance (OpenGL baseline)
- Headless compute: ~350fps
- With rendering (60-vertex instanced spheres): ~170fps
- With rendering (2-vertex lines): ~350fps (no impact)
- Bottleneck: fragment fill rate + overdraw, not vertex count
