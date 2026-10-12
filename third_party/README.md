# Dependency and dataset provenance

`dependency-manifest.json` identifies every bundled JAR by SHA-256 and links its
upstream project. Existing binaries are retained; none are fetched or rebuilt
during experiment setup. Package versions are encoded in filenames except the
two MomentSketch snapshots, whose originating source commits are unknown.

Apache Commons Math and Apache DataSketches JARs embed license/notice documents;
copies are preserved in their subdirectories here. MomentSketch, t-digest and
DataDog sketches retain their upstream licensing; source links and binary
identities are provided rather than treating them as part of the repository's
MIT-licensed original code. Verified upstream Apache 2.0 license texts are included
under `msketch/`, `t-digest/` and `sketches-java/`; these were fetched from the
respective upstream default branches during artifact preparation, not extracted
from those older binaries.

`GK.java` includes the original MIT notice for Lu Wang's quantile-alg
implementation. The Books downloader now uses an independent Python implementation
and retains the SOSD source URL/checksum instead of the formerly copied GPL shell
implementation. Dataset download manifests record accepted content hashes and
source URLs. UCI HEPMASS and Power are distributed under CC BY 4.0; cite the UCI
dataset pages when redistributing. Consult SOSD and its original Books data
sources before redistributing the Books binary; this artifact supplies a link
and download instructions rather than bundling it.
