.PHONY: all

SHELL=/usr/bin/env bash -eo pipefail

all: compile_SplineSketch compile_SplineSketchMG compile_SplineSketchAdjustable compile_tdigest compile_MS compile_KLL compile_GK compile_DDS

compile_SplineSketch:
	javac SplineSketchProgram.java SplineSketch.java

compile_SplineSketchMG:
	javac SplineSketchMGProgram.java SplineSketchMG.java

compile_SplineSketchLong:
	javac SplineSketchLongProgram.java SplineSketchLong.java

compile_SplineSketchAdjustable:
	javac SplineSketchAdjustableProgram.java SplineSketchAdjustable.java

compile_tdigest:
	javac -cp .:t-digest-3.3.jar TDigestProgram.java

compile_MS:
	javac -cp .:msolver-1.0-SNAPSHOT.jar:quantile-bench-1.0-SNAPSHOT.jar:commons-math3-3.6.1.jar MomentSketchProgram.java

compile_KLL:
	javac -cp .:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar KLLProgram.java

compile_GK:
	javac GKProgram.java GK.java

compile_DDS:
	javac -cp .:sketches-java-0.8.3.jar DDSketchProgram.java 
