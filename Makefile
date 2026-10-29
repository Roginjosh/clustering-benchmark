
# Clustering Benchmark
# Run "make help" to see available commands.

PYTHON ?= python3

DATASET ?= dtd
DATA_DIR ?= outputs/$(DATASET)
OUTPUT_DIR ?= outputs/benchmark

ALGORITHMS ?= kmeans gmm
REPRESENTATIONS ?= stem layer4
SEEDS ?= 0 1 2 3 4
THREADS ?= 4

RUNNER = $(PYTHON) -m scripts.experiments.run_benchmark

.PHONY: help test dtd dtd-full benchmark summary clean-results

.DEFAULT_GOAL := help

help:
	@echo ""
	@echo "Clustering Benchmark"
	@echo "===================="
	@echo ""
	@echo "  make test        Quick smoke test (64D, seed 42)"
	@echo "  make dtd         DTD 64D + 2048D, five seeds"
	@echo "  make dtd-full    DTD all five representations"
	@echo "  make benchmark   Custom benchmark using variables"
	@echo "  make summary     Summarize benchmark results"
	@echo ""
	@echo "Configuration:"
	@echo "  ALGORITHMS      $(ALGORITHMS)"
	@echo "  REPRESENTATIONS $(REPRESENTATIONS)"
	@echo "  SEEDS           $(SEEDS)"
	@echo "  THREADS         $(THREADS)"
	@echo ""
	@echo "Example:"
	@echo "  make benchmark ALGORITHMS=gmm REPRESENTATIONS=stem SEEDS=42"
	@echo ""

test:
	$(RUNNER) \
		--data-dir outputs/dtd \
		--dataset dtd \
		--representations stem \
		--algorithms kmeans gmm \
		--seeds 42 \
		--threads $(THREADS) \
		--output $(OUTPUT_DIR)/dtd_test.csv

dtd:
	$(RUNNER) \
		--data-dir outputs/dtd \
		--dataset dtd \
		--representations stem layer4 \
		--algorithms $(ALGORITHMS) \
		--seeds $(SEEDS) \
		--threads $(THREADS) \
		--output $(OUTPUT_DIR)/dtd_multiseed.csv

dtd-full:
	$(RUNNER) \
		--data-dir outputs/dtd \
		--dataset dtd \
		--representations stem layer1 layer2 layer3 layer4 \
		--algorithms $(ALGORITHMS) \
		--seeds $(SEEDS) \
		--threads $(THREADS) \
		--output $(OUTPUT_DIR)/dtd_full.csv

benchmark:
	$(RUNNER) \
		--data-dir $(DATA_DIR) \
		--dataset $(DATASET) \
		--representations $(REPRESENTATIONS) \
		--algorithms $(ALGORITHMS) \
		--seeds $(SEEDS) \
		--threads $(THREADS) \
		--output $(OUTPUT_DIR)/$(DATASET)_custom.csv

summary:
	$(PYTHON) -m scripts.experiments.summarize_benchmark \
		--input $(OUTPUT_DIR)/dtd_multiseed.csv \
		--output-csv $(OUTPUT_DIR)/dtd_summary.csv \
		--output-tex $(OUTPUT_DIR)/dtd_summary.tex

clean-results:
	@echo "To avoid accidental data loss, results are not deleted automatically."
	@echo "Inspect $(OUTPUT_DIR)/ and remove unwanted files manually."

