import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Arrays;
import java.util.Set;

/**
 * A Java translation of the SplineSketch Python code.
 * This includes:
 *  1) equallySpacedSelection
 *  2) mergeArrayIntoBuckets
 *  3) a SplineSketch class with similar logic
 *
 * Includes the PCHIP interpolation as a translation of scipy's PCHIP into Java.
 */
public class SplineSketch {

    // =============================================================
    // === Static functions
    // =============================================================

    /**
     * Equally spaced selection of k points from a sorted list.
     * This replicates the 'equally_spaced_selection' function from Python.
     *
     * @param lst Sorted list of doubles.
     * @param k   Number of points to select.
     * @return A List<Bucket> of length k, where each Bucket has:
     *         - threshold: the chosen boundary
     *         - count: 0  (since we haven't assigned counts yet)
     *         - isProtected: false
     */
    private static ArrayList<Object> equallySpacedSelection(double[] lst, int n, int k) {

        double step = (double)(n - 1) / (double)(k - 1);
        double[] thresholds = new double[k];
        for (int i = 0; i < k; i++) {
            int index = (int) Math.round(i * step);
            if (index >= n) {
                index = n - 1;
            }
            thresholds[i] = lst[index];
        }

        // "somewhat technical handling of repeated values of thresholds"
        int i = 0;
        while (i < k) {
            int j = i + 1;
            while (j < k && thresholds[i] >= thresholds[j]
                    - Math.abs(thresholds[j]) * 1e-9 - 1e-100) {
                j++;
            }
            if (j > i + 1) {
                double prev = (i > 0) ? thresholds[i - 1] : thresholds[0] - 1.0;
                double next = (j < k) ? thresholds[j] : thresholds[k - 1] + 1.0;
                thresholds[i] = thresholds[i] - (thresholds[i] - prev) * 1e-9;
                for (int jj = i + 2; jj < j; jj++) {
                    thresholds[jj] = thresholds[jj - 1] + (next - thresholds[jj - 1]) * 1e-3 / k;
                }
            }
            i = j;
        }

        // Check that thresholds are strictly increasing
        for (int idx = 0; idx < k - 1; idx++) {
            if (!(thresholds[idx] < thresholds[idx + 1])) {
                throw new AssertionError("Thresholds are not strictly increasing: " + thresholds);
            }
        }

        // Construct the (threshold, count, isProtected) triple but with count=0
        int[] counters = new int[k];
        boolean[] isProtected = new boolean[k];
        int j2 = 0;
        int prevJ = 0;
        for (int idx = 0; idx < k; idx++) {
            while (j2 < n && lst[j2] <= thresholds[idx]) {
                j2++;
            }
            //result.add(new Bucket(thresholds.get(idx), j2 - prevJ, false));
            counters[idx] = j2 - prevJ;
            isProtected[idx] = false;
            prevJ = j2;
        }
        ArrayList<Object> result = new ArrayList<>(3);
        result.add(thresholds);
        result.add(counters);
        result.add(isProtected);
        return result;
    }

    /**
     * Merges (sorted) array data into existing buckets. Only increments
     * the count for the first bucket in which array[idx_array] <= bucketThreshold.
     *
     * This replicates `merge_array_into_buckets` from Python.
     */
    private static void mergeArrayIntoBuckets(double[] array, int len, double[] thresholds, int[] counters, int  k) {
        int idxBuckets = 0;
        int idxArray = 0;

        while (idxArray < len && idxBuckets < k) {
            // While next array element is <= bucket[idxBuckets].threshold,
            // we add to that bucket's count.
            while (idxArray < len && thresholds[idxBuckets] >= array[idxArray]) {
                counters[idxBuckets]++;
                idxArray++;
            }
            idxBuckets++;
        }
    }

    // =============================================================
    // === The main SplineSketch class fields and methods
    // =============================================================

    private int k; // number of buckets
    private double[] thresholds;
    private int[] counters;
    private boolean[] isProtected;
    private double[] buffer;    // items that have not yet been consolidated into buckets
    private double[] errorEstimates;
    private double[] errorEstimatesAfterJoin;
    private double[] newBoundaries;
    private int[] prefSums;
    private double[] oldThresholds;
    private boolean[] newIsProtected;
    private long n;               // total count of items

    private String printInfo;

    private double splitJoinRatio;    // controls the comparison of “split vs. join” error estimates
    private int bufferSizeBound;      // maximum number of items in buffer before auto-consolidation
    private int bufferIndex;          // index of first free slot in the buffer

    private double minRelativeBucketLength;  // used in areSufficientlyDifferent
    private double minAbsoluteNonzeroValue;  // used for small numbers around zero

    private double minFracBucketBoundToSplit;

    private double epochIncrFactor;
    private long epochEnd;

    private double defaultBucketBoundMult;
    private double bucketBoundMult;

    /**
     * Constructs the SplineSketch with a specified number of buckets k.
     */
    public SplineSketch(int k, String printInfo) {
        if (k < 4) {
            throw new IllegalArgumentException("k must be >= 4");
        }
        this.k = k;
        this.thresholds = null; // new double[k];
        this.counters = null; // new int[k];
        this.isProtected = null; // new boolean[k];
        this.bufferSizeBound = 5 * k;
        this.buffer = new double[this.bufferSizeBound];
        this.errorEstimates = new double[k];
        this.errorEstimatesAfterJoin = new double[k];
        this.newBoundaries = new double[k];
        this.prefSums = new int[k];
        this.oldThresholds = new double[k];
        this.newIsProtected = new boolean[k];
        this.n = 0;
        this.printInfo = printInfo;

        // Constants (matching the Python code)
        this.splitJoinRatio = 1.5;
        this.minRelativeBucketLength = 1e-8;
        this.minAbsoluteNonzeroValue = Double.POSITIVE_INFINITY;
        this.minFracBucketBoundToSplit = 0.01;

        this.epochIncrFactor = 1.25;
        this.epochEnd = 2L * this.bufferSizeBound;
        this.defaultBucketBoundMult = 3.0;
        this.bucketBoundMult = 3.0;
    }

    /**
     * A convenience constructor when no debug info is needed.
     */
    public SplineSketch(int k) {
        this(k, "");
    }

    /**
     * This method returns an approximate number of bytes for storing the sketch
     * in an updatable form.
     */
    public int serializedSketchBytesUpdatable() {
        // k buckets => each bucket threshold is 8 bytes, count is 4 bytes,
        // plus we store a bit for isProtected
        // + 8 bytes for min_absolute_nonzero_value
        // Some minimal overhead for boolean array => ceiling(k/8).
        // These details match the Python code comments.
        return this.k * 16 + (int) Math.ceil(this.k / 8.0) + 8;
    }

    /**
     * Returns an approximate number of bytes for storing the sketch in a compact form.
     */
    public int serializedSketchBytesCompact() {
        // k thresholds => 8 bytes each, k counters => 8 bytes each, total 16*k,
        return this.k * 16;
    }

    /**
     * Checks whether two doubles a and b are "sufficiently different"
     * according to the logic in the Python code.
     */
    private boolean areSufficientlyDifferent(double a, double b) {
        double denom = Math.max(Math.max(Math.abs(a), Math.abs(b)), minAbsoluteNonzeroValue);
        double relDiff = Math.abs(a - b) / denom;
        return (relDiff > minRelativeBucketLength);
    }

    /**
     * Main consolidation routine (like Python's consolidate()).
     */
    public void consolidate() {
        // If no buffer and already k buckets, nothing to do.
        if (this.bufferIndex == 0 && this.thresholds.length == k) {
            return;
        }

        // If there are no existing buckets, just do an equally spaced selection
        // from the buffer.
        if (this.thresholds == null) {
            if (this.bufferIndex < k) { // too few items to create k buckets
                return;
            }
            Arrays.sort(buffer, 0, this.bufferIndex);
            List<Object> buckets = equallySpacedSelection(buffer, bufferIndex, k);
            this.thresholds = (double[])buckets.get(0);
            this.counters = (int[])buckets.get(1);
            this.isProtected = (boolean[])buckets.get(2);
            if (thresholds.length != k) {
                throw new AssertionError("The number of new buckets is " + thresholds.length + ", but k=" + k);
            }
            this.bufferIndex = 0;
            return;
        }

        // Possibly end an epoch => un-protect all buckets
        if ((long) n >= epochEnd) {
            for (int i = 0; i < this.isProtected.length; i++) {
                isProtected[i] = false;
            }
            epochEnd = (long) (epochIncrFactor * n);
            bucketBoundMult = defaultBucketBoundMult;
        }

        // Sort buffer -- TODO: faster using binsearch
        Arrays.sort(buffer, 0, this.bufferIndex);
        for (int i = 0; i < bufferIndex; i++) {
            if (buffer[i] != 0.0 && Math.abs(buffer[i]) < minAbsoluteNonzeroValue) { // moved to consolidate
                minAbsoluteNonzeroValue = Math.abs(buffer[i]);
            }
            if (buffer[i] > 0) break;
        }

        // Create a copy of the bucket thresholds
        int prefSum = 0;
        for (int i = 0; i < k; i++) {
            oldThresholds[i] = thresholds[i];
            prefSum += counters[i];
            prefSums[i] = prefSum;
        }

        // Merge buffer into that copy
        mergeArrayIntoBuckets(buffer, this.bufferIndex, thresholds, counters, k);

        double boundVal = bucketBoundMult * n / k; // note: multiplier may change during an iteration

        // int iter = 0;
        boolean performedChanges = true;
        while (performedChanges) {
            performedChanges = false;

            // Identify buckets that must be split
            // Condition 1: bucket count > 1.01 * bucketBound()
            // Condition 2: thresholds differ enough from the left neighbor
            Set<Integer> mustSplit = new HashSet<>();
            for (int i = 1; i < thresholds.length; i++) {
                if (counters[i] > 1.01 * boundVal &&
                        Math.abs(thresholds[i] - thresholds[i - 1]) / Math.max(Math.max(Math.abs(thresholds[i]), Math.abs(thresholds[i - 1])), minAbsoluteNonzeroValue) > minRelativeBucketLength) {
                    mustSplit.add(i);
                }
            }

            // Identify buckets that can be joined
            // Condition: sum of two adjacent bucket counts <= 0.75 * bound
            //            and not protected
            // We consider "i" as the left boundary to join with i+1
            List<Integer> canBeJoined = new ArrayList<>();
            for (int i = 1; i < thresholds.length - 1; i++) {
                double sumCount = counters[i] + counters[i + 1];
                if (sumCount <= 0.75 * boundVal && !isProtected[i]) {
                    canBeJoined.add(i);
                }
            }
            errorEstimates = computeErrorEstimates(thresholds, counters);
            errorEstimatesAfterJoin = computeErrorEstimatesAfterJoin(thresholds, counters);

            // Sort canBeJoined by the "join" error estimate
            canBeJoined.sort(Comparator.comparingDouble(i -> this.errorEstimatesAfterJoin[i]));

            // Filter out buckets that might be "relatively big" => candidate for splitting
            // Condition: difference in thresholds is large enough
            //            count > minFracBucketBoundToSplit * boundVal
            // Then sort descending by error estimate
            List<Integer> bucketsByErrors = new ArrayList<>();
            for (int i = 1; i < thresholds.length; i++) {
                if ((counters[i] > minFracBucketBoundToSplit * boundVal) &&
                    Math.abs(thresholds[i] - thresholds[i - 1]) / Math.max(Math.max(Math.abs(thresholds[i]), Math.abs(thresholds[i - 1])), minAbsoluteNonzeroValue) > minRelativeBucketLength) {
                    bucketsByErrors.add(i);
                }
            }
            bucketsByErrors.sort((a, b) -> Double.compare(errorEstimates[b], errorEstimates[a]));

            // Check if buffer extends beyond existing min/max => new extremes
            int newExtremes = 0;
            if (bufferIndex > 0) {
                double bufMin = buffer[0];
                double bufMax = buffer[bufferIndex - 1];
                if (bufMin < thresholds[0]) {
                    newExtremes++;
                    performedChanges = true;
                }
                if (bufMax > thresholds[thresholds.length - 1]) {
                    newExtremes++;
                    performedChanges = true;
                }
            }

            int resizeDiff = thresholds.length - k;

            // We'll also create sets for chosen splits/joins
            Set<Integer> willSplit = new HashSet<>(mustSplit);
            Set<Integer> willJoin = new HashSet<>();

            int idxJoin = 0;

            // We ensure we can join as many times as needed: each forced split or new extreme
            // might require a join to keep the size near k, etc.
            while (willJoin.size() < (mustSplit.size() + newExtremes + resizeDiff)
                    && idxJoin < canBeJoined.size()) {
                int candidate = canBeJoined.get(idxJoin);

                // check if neighbors are not already chosen
                if (!willJoin.contains(candidate) &&
                        !willJoin.contains(candidate - 1) &&
                        !willJoin.contains(candidate + 1) &&
                        !willSplit.contains(candidate) &&
                        !willSplit.contains(candidate + 1)) {
                    willJoin.add(candidate);
                }
                idxJoin++;
            }

            if (willJoin.size() < (mustSplit.size() + newExtremes + resizeDiff)) {
                // We cannot do anything => increase bucketBoundMult
                bucketBoundMult *= 2.0;
                boundVal *= 2.0;
                System.err.println("! willJoin.size() < (mustSplit.size() + newExtremes + resizeDiff) => increasing bucket bound multiplier to "
                        + bucketBoundMult + " at n=" + n + ", resize diff=" + resizeDiff
                        + ", epoch end=" + epochEnd + ", k=" + k
                        + " (info: " + printInfo + ")"); // iter=" + iter + ", 
                if (bucketBoundMult > 100) {
                    System.err.println("!!!!! RESETTING EPOCH !!!!"); // iter=" + iter + ", 
                    for (int i = 0; i < this.isProtected.length; i++) {
                        isProtected[i] = false;
                    }
                    epochEnd = (long) (epochIncrFactor * n);
                    bucketBoundMult = defaultBucketBoundMult;
                    boundVal = bucketBoundMult * n / k;
                }
                performedChanges = true;
                // iter++;
                continue; // proceed to next iteration
            }

            // Possibly add more splits if we are increasing k or if error is large
            int idxSplit = 0;
            int neededExtraSplits = -(willSplit.size() + newExtremes + resizeDiff);
            while (idxSplit < bucketsByErrors.size() && neededExtraSplits > 0) {
                int candidate = bucketsByErrors.get(idxSplit);
                if (!willSplit.contains(candidate)
                        && candidate != 0
                        && !willJoin.contains(candidate)
                        && !willJoin.contains(candidate - 1)) {
                    willSplit.add(candidate);
                    neededExtraSplits--;
                }
                idxSplit++;
            }

            // We also compare error estimates: if splitting helps more than joining
            while (idxJoin < canBeJoined.size() - (k / 4) - 2
                    && idxSplit < bucketsByErrors.size()) {
                int splitCandidate = bucketsByErrors.get(idxSplit);
                double splitError = errorEstimates[splitCandidate];

                int joinCandidate = canBeJoined.get(idxJoin);
                double joinError = errorEstimatesAfterJoin[joinCandidate];

                if (splitError > splitJoinRatio * joinError) {
                    // Check if this splitCandidate or joinCandidate is already used
                    if (!willSplit.contains(splitCandidate)
                            && splitCandidate != 0
                            && splitCandidate != joinCandidate
                            && splitCandidate != joinCandidate + 1
                            && !willJoin.contains(splitCandidate)
                            && !willJoin.contains(splitCandidate - 1)
                            && !willJoin.contains(joinCandidate)
                            && !willJoin.contains(joinCandidate + 1)
                            && !willSplit.contains(joinCandidate)
                            && !willSplit.contains(joinCandidate + 1)) {
                        willSplit.add(splitCandidate);
                        willJoin.add(joinCandidate);
                    }
                    idxSplit++;
                    idxJoin++;
                } else {
                    // If the split isn't bigger than ratio times the join => break
                    break;
                }
            }

            if (!willSplit.isEmpty() || !willJoin.isEmpty()) {
                performedChanges = true;
            }
            if (!performedChanges) {
                break;
            }

            // Rebuild the bucket boundaries
            int iB = 0;
            // Possibly add new extremes if buffer's min/max lie outside
            if (this.bufferIndex > 0) {
                double bufMin = buffer[0];
                double bufMax = buffer[bufferIndex - 1];
                if (bufMin < thresholds[0]) {
                    newBoundaries[0] = bufMin;
                    newIsProtected[0] = false;
                    iB++;
                }
                if (bufMax > thresholds[k - 1]) {
                    newBoundaries[k - 1] = bufMax;
                    newIsProtected[k - 1] = false;
                    //newmax++;
                }
            }

            boolean previousSplit = false;
            for (int i = 0; i < thresholds.length; i++) {
                // skip current boundary if we are joining it with the next
                if (willJoin.contains(i)) {
                    continue; // effectively merges i with i+1
                }

                // if we are splitting at i, add mid boundary between i-1 and i
                if (willSplit.contains(i)) {
                    // if (i == 0) {
                    //     throw new AssertionError("Should never split i=0 here.");
                    // }
                    double mid = 0.5 * (thresholds[i] + thresholds[i - 1]);
                    newBoundaries[iB] = mid;
                    // mark protected
                    if (iB > 0) {
                        newIsProtected[iB - 1] = true;
                    }
                    newIsProtected[iB] = true;
                    iB++;
                    previousSplit = true;
                }

                // add boundary i
                newBoundaries[iB] = thresholds[i];
                newIsProtected[iB] = isProtected[i] || previousSplit;
                iB++;
                previousSplit = false;
            }
            int prevVal = 0;
            int indOld = 0;
            for (int i = 0; i < k; i++) {
                double x = newBoundaries[i];
                thresholds[i] = x;
                // compute CDF value according to orig. buckets
                while (oldThresholds[indOld] < x && indOld < k - 1) { indOld++; }
                int currCDF;
                if (oldThresholds[indOld] <= x) {
                    currCDF = prefSums[indOld];
                } else {
                    currCDF = PchipInterpolator.EvalPCHIPatBucket(indOld, x, oldThresholds, prefSums);
                }
                // int currCDF = interpolator.valueAt(x);
                int cnt = currCDF - prevVal;
                counters[i] = cnt;
                isProtected[i] = newIsProtected[i];
                prevVal = currCDF;
            }

            mergeArrayIntoBuckets(buffer, this.bufferIndex, thresholds, counters, k);
        }

        // Clear the buffer
        bufferIndex = 0;
    }

    /**
     * Update the sketch with a single new item.
     */
    public void update(double item) {
        // If infinite or NaN => skip
        if (Double.isInfinite(item) || Double.isNaN(item)) {
            System.err.println("Cannot add " + item);
            return;
        }
        buffer[bufferIndex++] = item;
        n += 1;

        if (bufferIndex >= bufferSizeBound) {
            consolidate();
        }
    }

    /**
     * Returns a "PCHIP-like" spline interpolator over the buckets array
     * that, given x, returns the prefix sum up to x.
     *
     * In Python, we used SciPy's PchipInterpolator. Here, we provide
     * a placeholder that does a simple piecewise-linear approach. (You may
     * replace it with a real monotonic cubic spline if needed.)
     */
    private static PchipLikeInterpolator calcSpline(double[] thresholds, int[] counters) {
        if (thresholds == null) {
            return x -> 0;
        }
        // Construct prefix sums
        int[] prefixSums = new int[thresholds.length];
        int run = 0;
        for (int i = 0; i < thresholds.length; i++) {
            run += counters[i];
            prefixSums[i] = run;
        }

        return new PchipInterpolator(thresholds, prefixSums);
    }

    /**
     * Convenience method using current buckets.
     */
    private PchipLikeInterpolator calcSpline() {
        return calcSpline(this.thresholds, this.counters);
    }

    /**
     * Queries an array of items, returning approximate rank for each
     * (count of items <= that value).
     */
    public List<Integer> query(List<Double> items) {
        // Sort the buffer for local searching
        Arrays.sort(buffer, 0, bufferIndex);

        // For each item, find rank within buffer
        List<Integer> result = new ArrayList<>();
        PchipLikeInterpolator spline = calcSpline();
        for (int i = 0; i < items.size(); i++) {
            double val = items.get(i);
            // rank in buffer
            int rankBuffer = Arrays.binarySearch(buffer, 0, bufferIndex, val);
            if (rankBuffer < 0) {
                // insertion point is -(rankBuffer+1)
                rankBuffer = -rankBuffer - 1;
            } else {
                // If exact match, we want the position to the right
                // side='right' in Python means we count duplicates as well
                // Move forward while equals
                while (rankBuffer < this.bufferIndex && buffer[rankBuffer] <= val) {
                    rankBuffer++;
                }
            }

            int rankBuckets = 0;
            if (this.thresholds != null) {
                rankBuckets = spline.valueAt(val);
            }
            result.add(rankBuffer + rankBuckets);
        }
        return result;
    }
    
    /**
     * Computes the "error estimate" for bucket i, either for splitting or joining.
     */
    private double[] computeErrorEstimates(double[] thresholds, int[] counters) {
        double prevLen;
        int prevCnt;
        double nextLen = thresholds[1] - thresholds[0];
        int nextCnt = counters[1];
        int cnt = counters[0];
        double len = 1.0;
        double[] res = new double[thresholds.length];
        res[0] = 0.0;

        for (int i = 1; i < thresholds.length - 1; i++) {
            prevLen = len;
            prevCnt = cnt;
            len = nextLen;
            cnt = nextCnt;
            nextLen = thresholds[i + 1] - thresholds[i];
            nextCnt = counters[i + 1];

            double der2 = Math.max(
                    Math.abs((nextCnt / nextLen) - (cnt / len)) / (len + nextLen),
                    Math.abs((cnt / len) - (prevCnt / prevLen)) / (len + prevLen)
            );
            res[i] = (len * len) * Math.abs(der2);
        }
        
        prevLen = len;
        prevCnt = cnt;
        len = nextLen;
        cnt = nextCnt;
        nextLen = 1.0;
        nextCnt = 0;

        double der2 = Math.max(
                Math.abs((nextCnt / nextLen) - (cnt / len)) / (len + nextLen),
                Math.abs((cnt / len) - (prevCnt / prevLen)) / (len + prevLen)
        );
        res[thresholds.length - 1] = (len * len) * Math.abs(der2);
        return res;
    }

    /**
     * Computes the "error estimate" for bucket i, after joining with bucket i+1.
     */
    private double[] computeErrorEstimatesAfterJoin(double[] thresholds, int[] counters) {
        double prevLen;
        int prevCnt;
        int cnt = counters[0];
        double len = 1.0;
        double nextLen = thresholds[1] - thresholds[0];
        int nextCnt = counters[1];
        double nextNextLen = thresholds[2] - thresholds[1];
        int nextNextCnt = counters[2];
        double[] res = new double[thresholds.length];
        res[0] = 0.0;

        for (int i = 1; i < thresholds.length - 2; i++) {
            prevLen = len;
            prevCnt = cnt;
            len = nextLen;
            cnt = nextCnt;
            nextLen = nextNextLen;
            nextCnt = nextNextCnt;
            nextNextLen = thresholds[i + 2] - thresholds[i + 1];
            nextNextCnt = counters[i + 2];
            double currLen = len + nextLen;
            int currCnt = cnt + nextCnt;

            double der2 = Math.max(
                    Math.abs((nextNextCnt / nextNextLen) - (currCnt / currLen)) / (currLen + nextNextLen),
                    Math.abs((currCnt / currLen) - (prevCnt / prevLen)) / (currLen + prevLen)
            );
            res[i] = (currLen * currLen) * der2;
        }
        
        prevLen = len;
        prevCnt = cnt;
        len = nextLen;
        cnt = nextCnt;
        nextLen = nextNextLen;
        nextCnt = nextNextCnt;
        nextNextLen = 1.0;
        nextNextCnt = 0;

        double currLen = len + nextLen;
        int currCnt = cnt + nextCnt;

        double der2 = Math.max(
                Math.abs((nextNextCnt / nextNextLen) - (currCnt / currLen)) / (currLen + nextNextLen),
                Math.abs((currCnt / currLen) - (prevCnt / prevLen)) / (currLen + prevLen)
        );
        res[thresholds.length - 2] = (currLen * currLen) * der2;
        return res;
    }


    /**
     * A tiny functional interface for "CDF interpolation at x".
     */
    private interface PchipLikeInterpolator {
        int valueAt(double x);
    }

    /**
     * A Java implementation of 1D PCHIP interpolation
     * (Piecewise Cubic Hermite Interpolating Polynomial).
     */
    public static class PchipInterpolator implements PchipLikeInterpolator {

        private final double[] x;      // x-coordinates (strictly increasing)
        private final int[] y;      // y-coordinates
        private final double[] d;      // derivative at each x[i]

        // Note: assuming 0 < indBucket < thresholds.length - 1
        public static int EvalPCHIPatBucket(int indBucket, double x, double[] thresholds, int[] prefSums) {
            int k = thresholds.length;
            if (indBucket == 0) {
                if (x == thresholds[0])
                    return prefSums[0];
                else
                    return 0;
            }
            double[] localThr = new double[4];
            int[] localPrefSums = new int[4];
            localThr[1] = thresholds[indBucket - 1];
            localPrefSums[1] = prefSums[indBucket - 1];
            localThr[2] = thresholds[indBucket];
            localPrefSums[2] = prefSums[indBucket];
            if (indBucket > 1) {
                localThr[0] = thresholds[indBucket - 2];
                localPrefSums[0] = prefSums[indBucket - 2];
            } else {
                localThr[0] = thresholds[0] - 1.0;
                localPrefSums[0] = 0;
            }
            if (indBucket < k - 1) {
                localThr[3] = thresholds[indBucket + 1];
                localPrefSums[3] = prefSums[indBucket + 1];
            } else {
                localThr[3] = thresholds[k - 1] + 1.0;
                localPrefSums[3] = prefSums[k - 1];
            }
            PchipInterpolator interpolator = new PchipInterpolator(localThr, localPrefSums);
            return interpolator.valueAt(x);
        }

        /**
         * Constructs a PchipInterpolator given sorted x and corresponding y values.
         * 
         * @param x  strictly increasing array of x-values
         * @param y  array of y-values with same length as x
         * @throws IllegalArgumentException if input arrays are invalid
         */
        public PchipInterpolator(double[] x, int[] y) {
            this.x = x;
            this.y = y;

            // Compute the derivatives d[i] at each point using the PCHIP method:
            this.d = computeDerivatives(this.x, this.y);
        }

        /**
         * Compute the piecewise derivatives using the PCHIP algorithm.
         */
        private static double[] computeDerivatives(double[] x, int[] y) {
            final int n = x.length;
            double[] d = new double[n];

            // 1) Compute slopes of each interval: m_k = (y[k+1] - y[k]) / (x[k+1] - x[k])
            double[] mk = new double[n - 1];
            for (int k = 0; k < n - 1; k++) {
                mk[k] = (y[k + 1] - y[k]) / (x[k + 1] - x[k]);
            }

            // 2) For interior points k=1..n-2, compute derivative d[k].
            //    If mk[k] == 0 or mk[k-1] == 0 or they have opposite signs => d[k] = 0
            //    Otherwise, use the weighted harmonic mean.
            //    w1 = 2*h[k] + h[k-1], w2 = h[k] + 2*h[k-1]
            //    1/d[k] = (w1/mk[k-1] + w2/mk[k]) / (w1 + w2)
            double[] h = new double[n - 1];
            for (int k = 0; k < n - 1; k++) {
                h[k] = x[k + 1] - x[k];
            }

            // We'll fill in d[1..n-2] using the formula or zero
            for (int k = 1; k < n - 1; k++) {
                double mk0 = mk[k - 1];
                double mk1 = mk[k];

                // Check sign or zero:
                if (mk0 == 0.0 || mk1 == 0.0 || (mk0 > 0 && mk1 < 0) || (mk0 < 0 && mk1 > 0)) {
                    d[k] = 0.0;
                } else {
                    double w1 = 2.0 * h[k] + h[k - 1];
                    double w2 = h[k] + 2.0 * h[k - 1];
                    double denom = (w1 + w2);

                    // Weighted harmonic mean:
                    // whmean = (w1 / mk0 + w2 / mk1) / denom
                    // => d[k] = 1 / whmean
                    double numerator = (w1 / mk0) + (w2 / mk1);
                    double whmean = numerator / denom;
                    if (whmean == 0.0) {
                        d[k] = 0.0; // avoid Inf if something degenerate
                    } else {
                        d[k] = 1.0 / whmean;
                    }
                }
            }

            // 3) Handle endpoints d[0] and d[n-1] with the one-sided scheme.
            //    SciPy uses something akin to the method from:
            //    Cleve Moler, "Numerical Computing with MATLAB," Chap. 3.6 (pchiptx.m).
            d[0]     = edgeSlope(h[0], h[1], mk[0], mk[1]);
            d[n - 1] = edgeSlope(h[n - 2], h[n - 3], mk[n - 2], mk[n - 3]);

            return d;
        }

        /**
         * One-sided three-point estimate for the endpoint slope, with shape preservation.
         *
         * Similar to the _edge_case method in the Python code.
         */
        private static double edgeSlope(double h0, double h1, double m0, double m1) {
            // d = [(2*h0 + h1)*m0 - h0*m1] / (h0 + h1)
            double denom = (h0 + h1);
            if (denom == 0.0) {
                return 0.0; // degenerate spacing
            }
            double d = ((2.0 * h0 + h1) * m0 - h0 * m1) / denom;

            // Check sign mismatch with m0
            if (Math.signum(d) != Math.signum(m0)) {
                d = 0.0;
            }
            // If sign(m0) != sign(m1) and |d| > 3*|m0| => clamp
            if ((Math.signum(m0) != Math.signum(m1)) && (Math.abs(d) > 3.0 * Math.abs(m0))) {
                d = 3.0 * m0;
            }
            return d;
        }

        /**
         * Evaluate the interpolant at a given point X.
         * Uses piecewise-cubic Hermite polynomials on the interval containing X.
         *
         * @param X the query point
         * @return the interpolated function value at X
         */
        public int valueAt(double X) {
            final int n = x.length;
            // If X is out of range, either extrapolate or clamp to boundaries
            if (X <= x[0]) {
                return 0; // do not extrapolate left
            } else if (X >= x[n - 1]) {
                return y[n - 1]; // do not extrapolate right
            }

            // Binary search (or linear search) for interval i where x[i] <= X < x[i+1]
            int i = Arrays.binarySearch(x, X);
            if (i >= 0) {
                // X exactly matches x[i]
                return y[i];
            } else {
                // binarySearch returns -(insertion_point) - 1 if not found
                i = -i - 2;
            }
            // Evaluate on interval [i, i+1]
            return (int)hermiteInterpolate(i, X);
        }

        /**
         * Perform the actual cubic Hermite interpolation on the interval [i, i+1] for point X.
         * 
         * Using the standard Hermite basis:
         *     t = (X - x[i]) / (x[i+1] - x[i])
         *     h = x[i+1] - x[i]
         *
         *     H_1(t) =  2t^3 - 3t^2 + 1
         *     H_2(t) = -2t^3 + 3t^2
         *     H_3(t) =      t^3 - 2t^2 + t
         *     H_4(t) =      t^3 -   t^2
         *
         * p(t) = y[i]*H_1(t) + y[i+1]*H_2(t) + h*d[i]*H_3(t) + h*d[i+1]*H_4(t)
         *
         */
        private double hermiteInterpolate(int i, double X) {
            double h = x[i+1] - x[i];
            double t = (X - x[i]) / h;

            double t2 = t * t;
            double t3 = t2 * t;

            double h1 =  2.0 * t3 - 3.0 * t2 + 1.0;    // H_1(t)
            double h2 = -2.0 * t3 + 3.0 * t2;         // H_2(t)
            double h3 =         t3 - 2.0 * t2 + t;    // H_3(t)
            double h4 =         t3 -       t2;        // H_4(t)

            return y[i]     * h1
                + y[i+1]   * h2
                + (h * d[i])     * h3
                + (h * d[i+1])   * h4;
        }

        /**
         * Convenience method: evaluates the interpolant at multiple query points.
         *
         * @param X array of query points
         * @return array of interpolated values
         */
        public int[] values(double[] X) {
            int[] result = new int[X.length];
            for (int i = 0; i < X.length; i++) {
                result[i] = valueAt(X[i]);
            }
            return result;
        }

        // Getters for x, y, and d if desired, or other utility methods:
        public double[] getX() { return Arrays.copyOf(x, x.length); }
        public int[] getY() { return Arrays.copyOf(y, y.length); }
        public double[] getDerivatives() { return Arrays.copyOf(d, d.length); }
    }



    /**
     * A simple piecewise-linear interpolator that mimics the usage of PCHIP
     * only for cumulative distribution queries. The real PCHIP is more sophisticated,
     * but this is enough to demonstrate the general approach.
     */
    private static class LinearInterpolator implements PchipLikeInterpolator {
        private final double[] xs;
        private final int[] ys;

        public LinearInterpolator(double[] xs, int[] ys) {
            this.xs = xs;
            this.ys = ys;
        }

        @Override
        public int valueAt(double x) {
            if (xs.length == 0) {
                return 0;
            }
            if (x < xs[0]) {
                return 0;
            }
            if (x >= xs[xs.length - 1]) {
                return ys[ys.length - 1];
            }
            // find interval via binary search
            int idx = Arrays.binarySearch(xs, x);
            if (idx >= 0) {
                return ys[idx];
            } else {
                int insertPoint = -idx - 1;
                // linear interpolation between insertPoint-1 and insertPoint
                if (insertPoint == 0) {
                    return 0;
                }
                double x0 = xs[insertPoint - 1];
                double x1 = xs[insertPoint];
                int y0 = ys[insertPoint - 1];
                int y1 = ys[insertPoint];
                double ratio = (x - x0) / (x1 - x0);
                return (int)(y0 + ratio * (y1 - y0));
            }
        }
    }

    // =================================================================
    // === End of SplineSketch class
    // =================================================================
}
