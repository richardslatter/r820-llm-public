#!/bin/bash
# serve-best.sh: the LAN server on the best measured settings (build-cuda, dense split 0.5 with the VRAM switch,
# Q6_K DSpark draft, n_max 4)
cd "$(dirname "$0")"
exec env BIN=$HOME/src/ik_llama.cpp/build-cuda/bin GGML_CUDA_PIN_MAX_GB=60 LLAMA_OT_CPU_MMAP=1 LLAMA_F32_Q8=ffn_gate_inp \
  GGML_DENSE_GPU=1 GGML_VRAM_SWITCH=1 GGML_DENSE_GPU_RESERVE_MB=250 GGML_DENSE_GPU_FRAC=0.5 GGML_DENSE_GPU_FRACS=attn_kv.:0 \
  GGML_CUDA_DISABLE_GRAPHS=1 DRAFT=/mnt/models/dspark/dspark-ik-q6k.gguf NMAX=4 \
  ./serve.sh -ngl 0 -ngld 0 -cuda offload-batch-size=64 -b 2048 -ub 2048 -ctx-ckpt-t 0 "$@"
