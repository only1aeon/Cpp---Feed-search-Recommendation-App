#!/bin/bash
set -e
mkdir -p build
cd build
cmake ..
make -j$(nproc)
echo "Build complete. Run ./bin/recommender_demo"
