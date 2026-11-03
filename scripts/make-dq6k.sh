#!/bin/bash
# make-dq6k.sh SRC: the model on a RAM disk with its BF16 dense weights in Q6_K, in /mnt/models/dq6k.
# SRC holds the nine UD-Q8_K_XL shards of unsloth/DeepSeek-V4.1-Flash-1017-GGUF. Only the last shard holds BF16
# tensors, so the other eight are hard links. The requantiser builds against mainline llama.cpp's ggml
# (ML, default ~/src/llama.cpp; cloned and built if missing).
set -e
SRC=$1
ML=${ML:-$HOME/src/llama.cpp}
MLB=${MLB:-$ML/build/bin}
N=DeepSeek-V4.1-Flash-1017-UD-Q8_K_XL
D=/mnt/models
cd "$(dirname "$0")/.."

# huge=never: expert row quarters (1.6 MB) are smaller than a 2 MB page
sudo mkdir -p $D
mountpoint -q $D || sudo mount -t tmpfs -o size=560G,mpol=interleave,huge=never,uid=$(id -u),gid=$(id -g),mode=0755 tmpfs $D
mkdir -p $D/UD-Q8_K_XL $D/dq6k
cp -n "$SRC"/$N-0000?-of-00009.gguf $D/UD-Q8_K_XL/
for i in 1 2 3 4 5 6 7 8; do ln -f $D/UD-Q8_K_XL/$N-0000$i-of-00009.gguf $D/dq6k/; done

if [ ! -e $MLB/libggml-base.so ]; then
    if [ ! -d $ML ]; then
        git clone https://github.com/ggml-org/llama.cpp $ML && git -C $ML checkout -q f7c01c5226cac1de1052c17920ff38cdefea0ae6
    fi
    cmake -S $ML -B $ML/build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF && cmake --build $ML/build -j --target ggml-base
fi
gcc -O2 -march=native -fopenmp -I$ML/ggml/include tools/requant_shard.c -o requant_shard \
    -L$MLB -lggml-base -Wl,-rpath,$MLB
./requant_shard $D/UD-Q8_K_XL/$N-00009-of-00009.gguf $D/dq6k/$N-00009-of-00009.gguf q6_K token_embd.weight=keep
