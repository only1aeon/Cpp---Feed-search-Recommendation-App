        # C++ Recommender: Feed Ranking + Hybrid Search with FAISS

This repository contains a complete C++ implementation of the feed recommendation and hybrid search ranking system with FAISS-style vector search integration.

**What's included**
        - C++ code for hybrid search+feed ranking (semantic dense recall, ASR-aware lexical recall, cross-encoder placeholder).
        - FAISS integration (IndexFlatIP) for dense vector retrieval.
        - Modular code: src/, include/, cmake build (CMakeLists.txt), scripts to build/run.
        - Sample dataset and a small demo program.

**Notes**
        - This repo uses FAISS C++ library. Install FAISS (CPU) and make sure CMake finds it.
        - Cross-encoder and embedding inference are placeholders; replace with ONNXRuntime or other inference engine.

**Build (Linux)**
        ```
        # install dependencies (example for Ubuntu)
        sudo apt-get update
        sudo apt-get install -y build-essential cmake libblas-dev liblapack-dev
        # install faiss (e.g., pip wheel or from source) - make sure libfaiss is available for C++
        # install onnxruntime if you will run ONNX models
        mkdir build && cd build
        cmake ..
        make -j
        ./bin/recommender_demo
        ```
