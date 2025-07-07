#!/bin/bash

# Script to generate all permutations for TACO pretraining configs
# Usage: ./generate_all_permutations.sh

set -e  # Exit on any error

# Configuration
BASE_PATH="data_episodes/FullDataset/0.0"
OOD_CONSTRAINTS="basketball,bin-picking,button-press,push,shelf-place"
OUTPUT_DIR="configs_data_MT"
PERMUTATIONS=5

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "Generating TACO pretraining configurations..."
echo "OOD constraints: ${OOD_CONSTRAINTS}"
echo "Permutations per configuration: ${PERMUTATIONS}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Generate permutations for n_test = 10
echo "Generating configs for n_test=10..."
python generate_config.py \
    --base_path ${BASE_PATH} \
    --n_test 10 \
    --permutations ${PERMUTATIONS} \
    --ood_constraints "${OOD_CONSTRAINTS}" \
    --output ${OUTPUT_DIR}/config.json \
    --seed 42

echo ""

# Generate permutations for n_test = 20
echo "Generating configs for n_test=20..."
python generate_config.py \
    --base_path ${BASE_PATH} \
    --n_test 20 \
    --permutations ${PERMUTATIONS} \
    --ood_constraints "${OOD_CONSTRAINTS}" \
    --output ${OUTPUT_DIR}/config.json \
    --seed 43

echo ""

# Generate permutations for n_test = 40
echo "Generating configs for n_test=40..."
python generate_config.py \
    --base_path ${BASE_PATH} \
    --n_test 40 \
    --permutations ${PERMUTATIONS} \
    --ood_constraints "${OOD_CONSTRAINTS}" \
    --output ${OUTPUT_DIR}/config.json \
    --seed 44

echo ""

# Generate permutations for n_test = 49
echo "Generating configs for n_test=49..."
python generate_config.py \
    --base_path ${BASE_PATH} \
    --n_test 49 \
    --permutations ${PERMUTATIONS} \
    --ood_constraints "${OOD_CONSTRAINTS}" \
    --output ${OUTPUT_DIR}/config.json \
    --seed 45

echo ""
echo "All configurations generated successfully!"
echo "Check the ${OUTPUT_DIR} directory for all generated config files."

# List generated files
echo ""
echo "Generated files:"
ls -la ${OUTPUT_DIR}/*.json | awk '{print $9}' | sort
