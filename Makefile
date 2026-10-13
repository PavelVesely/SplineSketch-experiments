.PHONY: all setup doctor smoke reproduce plots paper test archive compile_SplineSketch compile_SplineSketchMG compile_SplineSketchAdjustable compile_tdigest compile_MS compile_KLL compile_GK compile_DDS

SHELL := /bin/bash
PYTHON ?= .venv/bin/python
PROFILE ?= paper
JAVAC ?= javac

all: compile_SplineSketch compile_SplineSketchMG compile_SplineSketchAdjustable compile_tdigest compile_MS compile_KLL compile_GK compile_DDS

compile_SplineSketch:
	$(JAVAC) --release 17 -encoding UTF-8 java/splinesketch/SplineSketchProgram.java java/splinesketch/SplineSketch.java

compile_SplineSketchMG:
	$(JAVAC) --release 17 -encoding UTF-8 java/splinesketch/SplineSketchMGProgram.java java/splinesketch/SplineSketchMG.java

compile_SplineSketchAdjustable:
	$(JAVAC) --release 17 -encoding UTF-8 java/splinesketch/SplineSketchAdjustableProgram.java java/splinesketch/SplineSketchAdjustable.java

compile_tdigest:
	$(JAVAC) --release 17 -encoding UTF-8 -cp java/other_sketches:t-digest-3.3.jar java/other_sketches/TDigestProgram.java

compile_MS:
	$(JAVAC) --release 17 -encoding UTF-8 -cp java/other_sketches:msolver-1.0-SNAPSHOT.jar:quantile-bench-1.0-SNAPSHOT.jar:commons-math3-3.6.1.jar java/other_sketches/MomentSketchProgram.java

compile_KLL:
	$(JAVAC) --release 17 -encoding UTF-8 -cp java/other_sketches:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar java/other_sketches/KLLProgram.java

compile_GK:
	$(JAVAC) --release 17 -encoding UTF-8 java/other_sketches/GKProgram.java java/other_sketches/GK.java

compile_DDS:
	$(JAVAC) --release 17 -encoding UTF-8 -cp java/other_sketches:sketches-java-0.8.3.jar java/other_sketches/DDSketchProgram.java

setup:
	python3 reproduce.py setup

doctor:
	$(PYTHON) reproduce.py doctor

smoke:
	$(PYTHON) reproduce.py run --profile smoke

reproduce:
	$(PYTHON) reproduce.py run --profile paper --download

plots:
	$(PYTHON) reproduce.py plot --output results/paper

paper:
	$(PYTHON) scripts/build_paper.py --profile $(PROFILE)

test:
	$(PYTHON) -m unittest discover -s tests -v

archive:
	$(PYTHON) scripts/package_artifact.py
