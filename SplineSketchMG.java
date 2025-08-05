import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.DoubleStream;

/**
 * A Java translation of the SplineSketch Python code.
 * This includes:
 *  1) equallySpacedSelection
 *  2) mergeIntoBuckets
 *  3) a SplineSketch class with similar logic
 *
 * Includes the PCHIP interpolation as a translation of scipy's PCHIP into Java.
 */
public class SplineSketchMG {

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
    private static ArrayList<Object> equallySpacedSelection(Map<Double, Integer> bufferFreqMap, int k) {
        double[] lst = bufferFreqMap.entrySet().stream()
                            .sorted(Map.Entry.comparingByKey())
                            .flatMapToDouble(entry -> 
                                DoubleStream.generate(() -> entry.getKey())
                                            .limit(entry.getValue()))
                            .toArray();
        int n = lst.length;
        // if (k > n) {
        //     throw new IllegalArgumentException("k cannot be greater than the length of the list");
        // }
        // if (k <= 1) {
        //     throw new IllegalArgumentException("k must be >= 2");
        // }

        double step = Math.max((double)(n - 1) / (double)(k - 1), 1.0);
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
            while (j < k && thresholds[i] >= thresholds[j] - Math.abs(thresholds[j]) * 1e-12 - 1e-100) {
                j++;
            }
            if (j > i + 1) {
                double prev = (i > 0) ? thresholds[i - 1] : thresholds[0] - Math.abs(thresholds[0]) - 1e-100;
                double next = (j < k) ? thresholds[j] : thresholds[k - 1] + Math.abs(thresholds[k - 1]) + 1e-100;
                thresholds[i] = thresholds[i] - Math.min((thresholds[i] - prev) / 2, Math.abs(thresholds[i]) * 1e-13 + 1e-100);
                for (int jj = i + 2; jj < j; jj++) {
                    thresholds[jj] = thresholds[jj - 1] + Math.min((next - thresholds[jj - 1]) / k, Math.abs(thresholds[jj - 1]) * 1e-13 + 1e-100);
                }
            }
            i = j;
        }

        // Check that thresholds are strictly increasing
        for (int idx = 0; idx < k - 1; idx++) {
            if (!(thresholds[idx] < thresholds[idx + 1])) {
                throw new AssertionError("Thresholds are not strictly increasing: thresholds[" + idx + "]=" + thresholds[idx] + ", thresholds[" + idx + "+1]=" + thresholds[idx + 1]);
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

    // /**
    //  * Merges (sorted) array data into existing buckets. Only increments
    //  * the count for the first bucket in which array[idx_array] <= bucketThreshold.
    //  *
    //  * This replicates `merge_array_into_buckets` from Python.
    //  */
    // private static void mergeArrayIntoBuckets(Map<Double, Integer> bufferFreqMap, double[] thresholds, int[] counters, int  k) {
    //     int idxBuckets = 0;
    //     int idxArray = 0;

    //     while (idxArray < len && idxBuckets < k) {
    //         // While next array element is <= bucket[idxBuckets].threshold,
    //         // we add to that bucket's count.
    //         while (idxArray < len && thresholds[idxBuckets] >= array[idxArray]) {
    //             counters[idxBuckets]++;
    //             idxArray++;
    //         }
    //         idxBuckets++;
    //     }
    // }

    /**
     * Merge the frequency map into bucket counters.
     * Assumes:
     * - thresholds is a sorted array.
     * - counters has length thresholds.length + 1.
     *   Bucket 0: x < thresholds[0]
     *   For i in 1..thresholds.length-1: thresholds[i-1] <= x < thresholds[i]
     *   Bucket thresholds.length: x >= thresholds[thresholds.length-1]
     */
    public static void mergeIntoBuckets(Map<Double, Integer> bufferFreqMap,
                                          double[] thresholds,
                                          int[] counters) {
        for (Map.Entry<Double, Integer> entry : bufferFreqMap.entrySet()) {
            double x = entry.getKey();
            int freq = entry.getValue();
            int bucket = findBucket(x, thresholds);
            // Increase the count for this bucket by the frequency
            if (bucket < thresholds.length)
                counters[bucket] += freq;
        }
    }

    /**
     * Determines the bucket index for x.
     * Uses Arrays.binarySearch to find the insertion point.
     * Returns:
     *   0 if x is less than thresholds[0];
     *   thresholds.length if x is greater than or equal to thresholds[thresholds.length-1];
     *   Otherwise, the index of the first threshold that is greater than x.
     */
    private static int findBucket(double x, double[] thresholds) {
        int idx = Arrays.binarySearch(thresholds, x);
        if (idx < 0) {
            // binarySearch returns (-(insertion point) - 1) if not found.
            idx = -idx - 1;
        }
        return idx;
    }

    // =============================================================
    // === The main SplineSketch class fields and methods
    // =============================================================

    private int k; // number of buckets
    //private List<Bucket> buckets;   // stored buckets
    private boolean updatable;
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

    private Map<Double, int[]> MGsketch;

    private String printInfo;

    private double splitJoinRatio;    // controls the comparison of “split vs. join” error estimates
    private int bufferSizeBound;      // maximum number of items in buffer before auto-consolidation
    private int bufferIndex;          // index of first free slot in the buffer

    private double minRelativeBucketLength;  // used in areSufficientlyDifferent
    // private int significantDigits;
    private double minAbsoluteNonzeroValue;  // used for small numbers around zero

    private double minFracBucketBoundToSplit;

    private double epochIncrFactor;
    private long epochEnd;

    private double defaultBucketBoundMult;
    private double bucketBoundMult;

    // For stats
    // private long totalIterCnt;
    // private long numConsolidates;

    /**
     * Constructs the SplineSketch with a specified number of buckets k.
     */
    public SplineSketchMG(int k, String printInfo) {
        if (k < 4) {
            throw new IllegalArgumentException("k must be >= 4");
        }
        this.k = k;
        this.updatable = true;
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

        this.MGsketch = new HashMap<>(k);

        // Constants (matching the Python code)
        this.splitJoinRatio = 1.5;
        this.minRelativeBucketLength = 1e-11;
        // this.significantDigits = 12;
        this.minAbsoluteNonzeroValue = Double.POSITIVE_INFINITY;
        this.minFracBucketBoundToSplit = 0.01;

        this.epochIncrFactor = 1.25;
        this.epochEnd = 2L * this.bufferSizeBound;
        this.defaultBucketBoundMult = 3.0;
        this.bucketBoundMult = 3.0;

        // Stats
        // this.totalIterCnt = 0;
        // this.numConsolidates = 0;
    }

    /**
     * A convenience constructor when no debug info is needed.
     */
    public SplineSketchMG(int k) {
        this(k, "");
    }

    /**
     * Returns the dynamic bucket capacity threshold, used in deciding splits/joins.
     */
    // private double bucketBound() {
    //     return bucketBoundMult * n / k;
    // }


    // public void printAvgItersStats() {
    //     double avg = (numConsolidates == 0) ? 0.0 : (double) totalIterCnt / (double) numConsolidates;
    //     System.out.printf("after %.0f updates, avg. num. iterations during consolidating is %f (%d / %d)\n",
    //             n, avg, totalIterCnt, numConsolidates);
    // }

    /**
     * This method returns an approximate number of bytes for storing the sketch
     * in an updatable form.
     */
    public int serializedSketchBytesUpdatable() {
        // k buckets => each bucket threshold is 8 bytes, count is 4 bytes,
        // plus we store a bit for isProtected
        // + 4 bytes for k, + 8 bytes for min_absolute_nonzero_value
        // Some minimal overhead for boolean array => ceiling(k/8).
        // These details match the Python code comments.
        // MG: k * (16 bytes for heavy hitters and their adjusted and true counters)
        return this.k * 40 + (int) Math.ceil(this.k / 8.0) + 12;
    }

    /**
     * Returns an approximate number of bytes for storing the sketch in a compact form.
     */
    public int serializedSketchBytesCompact() {
        // k thresholds => 8 bytes each, k counters => 8 bytes each, total 16*k,
        // plus 4 bytes for k.
        // MG: k * (16 bytes for heavy hitters and their true counters)
        return this.k * 16 + this.MGsketch.size() * 16;
    }

    // /**
    //  * Checks whether two doubles a and b are "sufficiently different"
    //  * according to the logic in the Python code.
    //  */
    // private boolean areSufficientlyDifferent(double a, double b) {
    //     double denom = Math.max(Math.max(Math.abs(a), Math.abs(b)), minAbsoluteNonzeroValue);
    //     double relDiff = Math.abs(a - b) / denom;
    //     return (relDiff > minRelativeBucketLength);
    // }

    /**
     * The non-heavy-hitters (freq. < n/k in MG will be merged into buckets)
     * and then if M heavy hitters remain in MG, they will be 
     */
    public void compressNonFrequentToBucketsAndResize() {
        // int nonHH = 0;
        // System.err.println("compressNonFrequentToBucketsAndResize");
        long thr = this.n / (2*this.k); // TODO: what is the right constant factor?
        this.updatable = false;
        List<Double> MGkeys = new ArrayList<>(MGsketch.keySet());
        for (int i = 0; i < MGkeys.size(); i++) {
            Double item = MGkeys.get(i);
            int[] freqs = MGsketch.get(item);
            int trueFreq = freqs[0];
            if (trueFreq < thr) {
                MGsketch.remove(item);
                // add the item with right multiplicity to the buffer
                for (int j = 0; j < trueFreq; j++) {
                    if (this.bufferIndex >= this.buffer.length) {
                        this.ensureBufferCapacity(2*this.bufferIndex);
                    }
                    buffer[bufferIndex++] = item;
                }
            }
        }
        // System.err.printf("MG size = %d, thresholds = %d, k = %d%n", MGsketch.size(), thresholds.length, k);

        int newK = Math.max(k - this.MGsketch.size(), Math.max(k / 2, 6)); // do not go below k/2 buckets
        if (newK != k)
            this.resize(newK);
        else
            this.consolidate();
        // System.err.printf("MG size = %d, thresholds = %d, k = %d%n", MGsketch.size(), thresholds.length, k);

    }
    /** Change the target bucket count (k) and immediately re-consolidate. */
    public void resize(int newK) {
        if (newK < 6) {
            throw new IllegalArgumentException("new_k must be ≥ 6");
        }
        if (newK != this.k) {
            // A “large” change => force a new epoch just like in Python
            if (Math.abs(this.k - newK) > 0.25 * this.k) {
                this.epochEnd = 0;
            }
            /* update buffer bounds and all aux arrays */
            ensureBufferCapacity(bufferIndex);           // keep current data
            ensureAuxArraysSize(Math.max(this.k, newK));
            this.k = newK;
            this.bufferSizeBound = 5 * this.k;


            consolidate();                               // bring sketch to size k
        }
    }

    /** Ensure the buffer array is large enough (doubling strategy). */
    private void ensureBufferCapacity(int needed) {
        if (this.buffer.length < needed) {
            int newLen = Math.max(needed, this.buffer.length * 2);
            this.buffer = Arrays.copyOf(this.buffer, newLen);
        }
    }

    /** Resize *all* auxiliary arrays that depend on k if <code>size</code> is larger. */
    private void ensureAuxArraysSize(int size) {
        if (this.errorEstimates.length == size) return;   // nothing to do
        this.errorEstimates         = new double[size];
        this.errorEstimatesAfterJoin= new double[size];
        this.newBoundaries          = new double[size];
        this.prefSums               = new int[size];
        this.oldThresholds          = new double[size];
        this.newIsProtected         = new boolean[size];
    }

    
    /**
     * Main consolidation routine (like Python's consolidate()).
     */
    public void consolidate() {
        // If no buffer and already k buckets, nothing to do.
        if (this.bufferIndex == 0 && (this.thresholds == null || this.thresholds.length == k)) {
            return;
        }

        // Count frequency of each number
        Map<Double, Integer> bufferFreqMap = new HashMap<>();
        for (int i = 0; i < bufferIndex; i++) {
            double num = buffer[i];
            int[] freqs = MGsketch.getOrDefault(num, null); // TODO: can be moved to the for cycle below
            if (freqs != null) {
                freqs[0]++;
                freqs[1]++; 
            } else {
                bufferFreqMap.put(num, bufferFreqMap.getOrDefault(num, 0) + 1);
            }
        }
        this.bufferIndex = 0;

        // // Convert array to list for sorting
        List<Double> bufferList = new ArrayList<>(bufferFreqMap.keySet());

        // // Sort based on frequency (descending) -- not clear if it helps... TODO: try it
        // bufferList.sort((a, b) -> {
        //     return bufferFreqMap.get(b).compareTo(bufferFreqMap.get(a)); // Descending frequency
        //     // int freqCompare = bufferFreqMap.get(b).compareTo(bufferFreqMap.get(a)); // Descending frequency
        //     // return freqCompare != 0 ? freqCompare : Double.compare(a, b); // Ascending value
        // });

        
        // for (Map.Entry<Double, Integer> bufEntry : bufferFreqMap.entrySet()) { // does not work as we'd like to remove stuff from the map
        if (updatable) {
            for (int j = 0; j < bufferList.size(); j++) {
                Double key = bufferList.get(j);
                int freq = bufferFreqMap.get(key);
                if (MGsketch.size() < k) { // new item does not fit
                    MGsketch.put(key, new int[]{freq, freq});
                    bufferFreqMap.remove(key);
                }
                else {
                    int minFreq = Integer.MAX_VALUE;
                    for (Map.Entry<Double, int[]> entry : MGsketch.entrySet()) {
                        int freq2 = entry.getValue()[0];
                        if (freq2 < minFreq) {
                            minFreq = freq2;
                        }
                    }
                    minFreq = Math.min(minFreq, freq);
                    boolean added = false;
                    // for (Map.Entry<Double, int[]> entry : MGsketch.entrySet()) { // does not work as we'd like to remove stuff from MG
                    List<Double> MGkeys = new ArrayList<>(MGsketch.keySet());
                    for (int i = 0; i < MGkeys.size(); i++) {
                        Double keyMG = MGkeys.get(i);
                        int[] freqs = MGsketch.get(keyMG);
                        freqs[1] -= minFreq;
                        if (freqs[1] == 0) {
                            MGsketch.remove(keyMG);
                            bufferFreqMap.put(keyMG, freqs[0]);
                            if (!added) {
                                MGsketch.put(key, new int[]{freq, freq});
                                bufferFreqMap.remove(key);
                                added = true;
                            }
                        }
                    }
                }
            }
        }

        // If there are no existing buckets, just do an equally spaced selection
        // from the buffer.
        if (this.thresholds == null) {
            if (bufferFreqMap.size() == 0)
                return;
            List<Object> buckets = equallySpacedSelection(bufferFreqMap, k);
            this.thresholds = (double[])buckets.get(0);
            this.counters = (int[])buckets.get(1);
            this.isProtected = (boolean[])buckets.get(2);
            if (thresholds.length != k) {
                throw new AssertionError("The number of new buckets is " + thresholds.length + ", but k=" + k);
            }
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

        // TODO: faster using binsearch
        double bufMin = Double.POSITIVE_INFINITY;
        double bufMax = Double.NEGATIVE_INFINITY;
        for (Double item : bufferFreqMap.keySet()) {
            if (item != 0.0 && Math.abs(item) < minAbsoluteNonzeroValue) { // moved to consolidate
                minAbsoluteNonzeroValue = Math.abs(item);
            }
            if (item < bufMin) bufMin = item;
            if (item > bufMax) bufMax = item;
        }

        
        int currNumThresholds = thresholds.length; // needed for merging or resizing
        // double[] newThresholds = this.thresholds;
        // int[] newCounters = Arrays.copyOf(this.counters, this.counters.length);
        // boolean[] newIsProtected = this.isProtected;

        // Build an interpolator from old buckets -- before merging in the buffer
        //PchipLikeInterpolator interpolator = calcSpline(this.thresholds, this.counters); // TODO: possibly not needed many times; or we can just use local interpolation
        int prefSum = 0;
        for (int i = 0; i < currNumThresholds; i++) {
            oldThresholds[i] = thresholds[i];
            prefSum += counters[i];
            prefSums[i] = prefSum;
        }

        mergeIntoBuckets(bufferFreqMap, thresholds, counters);

        

        int iter = 0;
        boolean performedChanges = true;
        while (performedChanges) {
            performedChanges = false;

			double boundVal = bucketBoundMult * n / k; // note: multiplier may change during an iteration
            // Identify buckets that must be split
            // Condition 1: bucket count > 1.01 * bucketBound()
            // Condition 2: thresholds differ enough from the left neighbor
            Set<Integer> mustSplit = new HashSet<>();
            for (int i = 1; i < currNumThresholds; i++) {
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
            for (int i = 1; i < currNumThresholds - 1; i++) {
                int sumCount = counters[i] + counters[i + 1];
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
            for (int i = 1; i < currNumThresholds; i++) {
                if ((counters[i] > minFracBucketBoundToSplit * boundVal) &&
                    Math.abs(thresholds[i] - thresholds[i - 1]) / Math.max(Math.max(Math.abs(thresholds[i]), Math.abs(thresholds[i - 1])), minAbsoluteNonzeroValue) > minRelativeBucketLength) {
                    bucketsByErrors.add(i);
                }
            }
            bucketsByErrors.sort((a, b) -> Double.compare(errorEstimates[b], errorEstimates[a]));

            // Check if buffer extends beyond existing min/max => new extremes
            int newExtremes = 0;
            if (bufferFreqMap.size() > 0) {
                if (bufMin < thresholds[0]) {
                    newExtremes++;
                    performedChanges = true;
                }
                if (bufMax > thresholds[currNumThresholds - 1]) {
                    newExtremes++;
                    performedChanges = true;
                }
            }

            int resizeDiff = currNumThresholds - k;

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

            if ((resizeDiff == 0 || iter > 0) && willJoin.size() < (mustSplit.size() + newExtremes + resizeDiff)) {
                // We cannot do anything => increase bucketBoundMult
                bucketBoundMult *= 2.0;
                boundVal *= 2.0;
                System.err.println("! willJoin.size() < (mustSplit.size() + newExtremes + resizeDiff) => increasing bucket bound multiplier to "
                        + bucketBoundMult + " at n=" + n + ", resize diff=" + resizeDiff
                        + ", epoch end=" + epochEnd + ", k=" + k
                        + " (info: " + printInfo + ")"); // iter=" + iter + ", 
                if (bucketBoundMult > 100) {
                    System.err.println("!!!!! RESETTING EPOCH !!!!"); // iter=" + iter + ", 
                    for (int i = 0; i < currNumThresholds; i++) {
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
            // List<Double> newBoundaries = new ArrayList<>();
            // List<Boolean> newIsProtected = new ArrayList<>();
            int iB = 0;
            //int newmax = 0; //TODO: tmp
            // Possibly add new extremes if buffer's min/max lie outside
            if (bufferFreqMap.size() > 0) {
                if (bufMin < thresholds[0]) {
                    newBoundaries[0] = bufMin;
                    newIsProtected[0] = false;
                    iB++;
                }
            }

            boolean previousSplit = false;
            for (int i = 0; i < currNumThresholds; i++) {
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
            if (bufferFreqMap.size() > 0) {
                if (bufMax > thresholds[currNumThresholds - 1]) {
                    newBoundaries[iB] = bufMax;
                    newIsProtected[iB] = false;
                    iB++;
                }
            }
            // if (iB + newmax != k) {
            //     //throw new AssertionError(
            //         System.err.println("!!! iB=" + iB + ", newmax = " + newmax + ", k=" + k);
            // }

            // Ensure strictly increasing boundaries
            // for (int iB = 0; iB < newBoundaries.size() - 1; iB++) {
            //     if (newBoundaries.get(iB) >= newBoundaries.get(iB + 1)) {
            //         throw new AssertionError("New boundaries are not strictly increasing!");
            //     }
            // }

            // Recompute bucket counts by interpolation and then merging the buffer
            //List<Bucket> updated = new ArrayList<>(newBoundaries.size());
            // thresholds = new double[newBoundaries.size()];
            // counters = new int[newBoundaries.size()];
            // newIsProtected = new boolean[newBoundaries.size()];
            int prevVal = 0;
            int indOld = 0;
            currNumThresholds = iB;
            if (thresholds.length != currNumThresholds) {
                thresholds = new double[currNumThresholds];
                counters = new int[currNumThresholds];
                isProtected = new boolean[currNumThresholds];
            }
            for (int i = 0; i < currNumThresholds; i++) {
                double x = newBoundaries[i];
                thresholds[i] = x;
                // compute CDF value according to orig. buckets
                while (oldThresholds[indOld] < x && indOld < oldThresholds.length - 1) { indOld++; }
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

            mergeIntoBuckets(bufferFreqMap, thresholds, counters);

            iter++;
        }
        assert currNumThresholds == k;
        if (thresholds.length > k) {
            double[] newThr = new double[currNumThresholds];
            System.arraycopy(thresholds, 0, newThr, 0, currNumThresholds);
            thresholds = newThr;
            int[] newCntrs = new int[currNumThresholds];
            System.arraycopy(counters, 0, newCntrs, 0, currNumThresholds);
            counters = newCntrs;
            boolean[] newProt = new boolean[currNumThresholds];
            System.arraycopy(isProtected, 0, newProt, 0, currNumThresholds);
            isProtected = newProt;
        }
        assert thresholds.length == k;

        // this.thresholds = newThresholds;
        // this.counters = newCounters;
        // this.isProtected = newIsProtected;

        // Double-check sum of counts //TODO
        int totalCount = 0;
        for (int i = 0; i < k; i++) {
            totalCount += counters[i];
        }
        for (Map.Entry<Double,int[]> entry : MGsketch.entrySet()) {
            totalCount += entry.getValue()[0];
        }
        if (Math.abs(totalCount - n) > 0.1) {
            throw new AssertionError(
                "Bucket counts do not sum to n: sum=" + totalCount + ", n=" + n);
        }

        // Update stats
        // totalIterCnt += iter;
        // numConsolidates++;
    }

    public static double roundToSignificantDigits(double num, int signifDigits) {
        if (num == 0) return 0;
        final int power = signifDigits - (int)Math.ceil(Math.log10(Math.abs(num)));
        final double magnitude = Math.pow(10, power);
        return Math.round(num * magnitude) / magnitude;
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
        buffer[bufferIndex++] = item; //roundToSignificantDigits(item, this.significantDigits);
        n += 1;

        // if (Math.abs(item) > 0.0 && Math.abs(item) < minAbsoluteNonzeroValue) { // moved to consolidate
        //     minAbsoluteNonzeroValue = Math.abs(item);
        // }

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
            double val = items.get(i); //roundToSignificantDigits(items.get(i), this.significantDigits);
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
            int rankMG = 0;
            for (Map.Entry<Double,int[]> entry : MGsketch.entrySet()) {
                if (entry.getKey() <= val) {
                    rankMG += entry.getValue()[0];
                }
            }

            int rankBuckets = 0;
            if (this.thresholds != null) {
                rankBuckets = spline.valueAt(val);
            }
            result.add(rankBuffer + rankBuckets + rankMG);
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
        double len = nextLen;
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
            res[i] = (len * len) * der2;
        }
        
        prevLen = len;
        prevCnt = cnt;
        len = nextLen;
        cnt = nextCnt;
        nextCnt = 0;

        double der2 = Math.max(
                Math.abs((nextCnt / nextLen) - (cnt / len)) / (len + nextLen),
                Math.abs((cnt / len) - (prevCnt / prevLen)) / (len + prevLen)
        );
        res[thresholds.length - 1] = (len * len) * der2;
        return res;
    }

    /**
     * Computes the "error estimate" for bucket i, after joining with bucket i+1.
     */
    private double[] computeErrorEstimatesAfterJoin(double[] thresholds, int[] counters) {
        double prevLen;
        int prevCnt;
        int cnt = counters[0];
        double nextLen = thresholds[1] - thresholds[0];
        double len = nextLen;
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


    // =================================================================
    // === End of SplineSketch class
    // =================================================================
}
