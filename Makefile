PYTHON ?= python3
ARCH ?= x86-64
COMP ?= gcc
JOBS ?= 2
EXE ?= stockless-variants
BUILD_DIR ?= build/engine

.PHONY: all materialize build test catalog package
all: build

materialize:
	$(PYTHON) tools/materialize.py --output $(BUILD_DIR)

build: materialize
	$(MAKE) -C $(BUILD_DIR)/src -j$(JOBS) build ARCH=$(ARCH) COMP=$(COMP) largeboards=yes all=yes nnue=no EXE=$(EXE)

test:
	$(PYTHON) tests/run_engine_tests.py --engine $(BUILD_DIR)/src/$(EXE) --report build/engine-tests.json

catalog:
	$(PYTHON) tools/catalog_variants.py --engine $(BUILD_DIR)/src/$(EXE)

package:
	$(PYTHON) tools/package.py --engine $(BUILD_DIR)/src/$(EXE) --build-dir $(BUILD_DIR) --arch $(ARCH)
