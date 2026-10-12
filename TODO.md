# Author follow-up

- [ ] Locate the GKAdaptive implementation used for the published Figure 8
  (maximum merge error on Normal and HEPMASS, and merge time on Normal).
  `GKProgram.java` in this checkout has no merge implementation and previously
  ignored the optional merge partition argument. Confirm whether the published
  curves used a different implementation or streaming measurements. Supply the
  original code and measurement protocol before enabling the GKAdaptive merge
  curve. The reproduction runner currently includes SplineSketch, KLL,
  MomentSketch, and t-digest for this figure and labels the missing GKAdaptive
  merge curve.

- [ ] Publish a versioned artifact archive at a persistent public URL (for example
  a Zenodo DOI), then add that URL to the SIGMOD ARI submission and this README.
  Include the source, bundled dependencies and their notices, configurations,
  documentation, and validation report. Dataset URLs and download scripts are
  included; check upstream redistribution conditions before bundling datasets.
