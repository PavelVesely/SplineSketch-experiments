.PHONY: all compile_java

SHELL=/usr/bin/env bash -eo pipefail

all: compile_java

compile_java:
	echo "compiling SplineSketch"
	javac SplineSketchProgram.java SplineSketch.java
	echo "compiling t-digest"
	javac -cp .:t-digest-3.3.jar TDigestProgram.java
	echo "compiling MomentSketchProgram"
	javac -cp .:msolver-1.0-SNAPSHOT.jar:quantile-bench-1.0-SNAPSHOT.jar:commons-math3-3.6.1.jar MomentSketchProgram.java
	echo "compiling KLLProgram"
	javac -cp .:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar KLLProgram.java
	echo "compiling ReqSketchProgram"
	javac -cp .:datasketches-java-6.0.0.jar:datasketches-memory-2.2.1.jar ReqSketchProgram.java

