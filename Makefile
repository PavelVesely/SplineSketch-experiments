.PHONY: all setup doctor smoke reproduce plots test archive compile_SplineSketch compile_SplineSketchMG compile_SplineSketchAdjustable compile_tdigest compile_MS compile_KLL compile_GK compile_DDS

SHELL := /bin/bash
PYTHON ?= .venv/bin/python
JAVAC ?= javac

all: compile_SplineSketch compile_SplineSketchMG compile_SplineSketchAdjustable compile_tdigest compile_MS compile_KLL compile_GK compile_DDS

compile_SplineSketch:
	$(JAVAC) -encoding UTF-8 SplineSketchProgram.java SplineSketch.java

compile_SplineSketchMG:
	$(JAVAC) -encoding UTF-8 SplineSketchMGProgram.java SplineSketchMG.java

compile_SplineSketchAdjustable:
	$(JAVAC) -encoding UTF-8 SplineSketchAdjustableProgram.java SplineSketchAdjustable.java

compile_tdigest:
	$(JAVAC) -encoding UTF-8 -cp .:t-digest-3.3.jar TDigestProgram.java

compile_MS:
	$(JAVAC) -encoding UTF-8 -cp .:msolver-1.0-SNAPSHOT.jar:quantile-bench-1.0-SNAPSHOT.jar:commons-math3-3.6.1.jar MomentSketchProgram.java

compile_KLL:
	$(JAVAC) -encoding UTF-8 -cp .:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar KLLProgram.java

compile_GK:
	$(JAVAC) -encoding UTF-8 GKProgram.java GK.java

compile_DDS:
	$(JAVAC) -encoding UTF-8 -cp .:sketches-java-0.8.3.jar DDSketchProgram.java

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

test:
	$(PYTHON) -m unittest discover -s tests -v

archive:
	$(PYTHON) scripts/package_artifact.py
