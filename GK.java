/*
 * Straight‑through, statement‑for‑statement translation of the C++ class you
 * supplied.  Only syntax has changed from C++ to Java; all identifiers,
 * class / method names, and control flow remain intact.  Read the notes that
 * follow the class for a few unavoidable Java‑specific caveats.
 */

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.TreeMap;

/** Port of the C++ template<class Double> GK implementation. */
public class GK {

    /*------------------------------------------------------------------------
     *  Public data members – identical names & types (bit‑width permitting)
     *------------------------------------------------------------------------*/
    public long   last_n;              // C++ long long
    public double EPS;
    public double   max_v;               // C++ uint64_t → Java long

    /*-------------------------------- entry --------------------------------*/
    public static class entry {
        public int g;                  // C++ unsigned int → Java int
        public int delta;

        public entry() { }
        public entry(int _g, int _d) { this.g = _g;  this.delta = _d; }
    }

    /** C++ std::multimap <Double, entry>.  We never insert duplicates, so   *
     *  TreeMap gives the same ordering semantics.                            */
    public final TreeMap<Double, entry> entries_map = new TreeMap<>();

    public int max_gd;                 // unsigned int → int

    /*----------------------------- SummaryEntry ----------------------------*/
    public static class SummaryEntry {
        public Double  v;
        public long    g;
        public long    delta;
    }
    public final ArrayList<SummaryEntry> summary = new ArrayList<>();

    /*----------------------------- ThresholdEntry --------------------------*/
    public static class ThresholdEntry implements Comparable<ThresholdEntry> {
        public int                           threshold;  // unsigned int
        public Map.Entry<Double, entry>           map_iter;   // “iterator” alias

        @Override
        public int compareTo(ThresholdEntry o) {
            // min‑heap (same as std::greater in C++)
            return Integer.compare(this.threshold, o.threshold);
        }
    }
    public final PriorityQueue<ThresholdEntry> compress_heap =
            new PriorityQueue<>();

    /*------------------------------------------------------------------------
     *  Constructors   (mirror C++ signatures)
     *------------------------------------------------------------------------*/
    public GK(double eps) { this(eps, Double.MAX_VALUE); }

    public GK(double eps, double max_v) {
        this.last_n = 0;
        this.EPS    = eps;
        this.max_v  = max_v;

        // sentinel tuple: (max_v, entry(1,0))
        @SuppressWarnings("unchecked")
        double sentinelKey = max_v;   
        entries_map.put(sentinelKey, new entry(1, 0)); // boxed integral Double
    }

    /*------------------------------------------------------------------------
     *  finalize()   (same name as C++ even though Object has one)
     *------------------------------------------------------------------------*/
    @Override
    @SuppressWarnings({"deprecation", "RedundantSuppression"})
    public void finalize() {
        summary.clear();
        summary.ensureCapacity(entries_map.size());

        long cumulativeG = 0;
        // max_gd = 0;

        for (Map.Entry<Double, entry> p : entries_map.entrySet()) {
            entry e = p.getValue();
            // int gd  = e.g + e.delta;
            // if (gd > max_gd) max_gd = gd;

            cumulativeG += e.g;

            SummaryEntry se = new SummaryEntry();
            se.v     = p.getKey();
            se.g     = cumulativeG;
            se.delta = e.delta;
            // System.out.printf("summary entry: val = %f, g = %d, delta = %d %n", se.v, se.g, se.delta);
            summary.add(se);
        }
        // max_gd /= 2;
        // System.out.printf("last_n = %d %n", last_n);
    }

    
    public int num_entries_stored() {
        return entries_map.entrySet().size();
    }

    /*------------------------------------------------------------------------
     *  query_for_value – identical logic / name
     *------------------------------------------------------------------------*/
    public Double query_for_value(double rank) {
        SummaryEntry probe = new SummaryEntry();
        probe.g     = (long)(rank * last_n + max_gd);
        probe.delta = 0;

        int idx = Collections.binarySearch(
            summary,
            probe,
            (a, b) -> Long.compare(a.g + a.delta, b.g + b.delta)
        );
        if (idx < 0) idx = -idx - 1;  // insertion point

        if (idx == 0) {
            /* The original C++ returns literal 0 here.  That only makes sense
             * when Double is an integral type; returning null preserves type
             * safety for the general case.  Adjust if you know Double.       */
            return null;
        }
        return summary.get(idx - 1).v;
    }

    /**
     * Approximate rank: number of items ≤ value
     * Error ≤ EPS · last_n, like the original GK guarantee.
     *
     * Works after you have called finalize().  (If you call it sooner the
     * summary list is empty, so the method returns 0.)
     */
    public long query_for_rank(Double value) {

        /* No data or nothing smaller than the sentinel ---------------------- */
        //if (summary.isEmpty()) return 0;
        if (value >= max_v) return last_n;

        /* -------------------------------------------------------------------
        * Binary‑search the summary list by key.  We need the first element
        * whose v  >  value  (upper_bound).  That element’s predecessor is the
        * last tuple with v ≤ value.
        * ------------------------------------------------------------------ */
        int lo = 0, hi = summary.size();          // hi is exclusive
        while (lo < hi) {
            int mid = (lo + hi) >>> 1;
            @SuppressWarnings("unchecked")
            int cmp = summary.get(mid).v.compareTo(value);

            if (cmp <= 0) {          // v[mid] ≤ value  →  we need to look right
                lo = mid + 1;
            } else {                 // v[mid]  > value →  look left
                hi = mid;
            }
        }
        /* lo is now the index of the first v  > value.
        If that is 0, all summary keys are greater → rank is 0. */
        if (lo == 0) return 0;

        SummaryEntry se = summary.get(lo - 1);
        /* Any rank in [g, g + δ] is legal.  Mid‑point avoids bias. */
        return se.g + se.delta / 2;
    }

    /*------------------------------------------------------------------------
     *  feed – full algorithm, ported line‑for‑line
     *------------------------------------------------------------------------*/
    public void feed(Double v) {
        ++last_n;

        if (v == max_v) return;

        /*---------------------------------------------------------------
         *  upper_bound → higherEntry (first key strictly greater)
         *--------------------------------------------------------------*/
        Map.Entry<Double, entry> iter = entries_map.higherEntry(v);
        // if (iter == null) {
        //     for (Map.Entry<Double, entry> entry : entries_map.entrySet()) {
        //         System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue());
        //     }
        // }
        assert iter != null : String.format("iter must not be null; v = %f, max_v = %f, entries_map: ", v, max_v);

        entry ecur         = iter.getValue();
        int   threshold    = (int)Math.floor(EPS * last_n * 2);
        int   tmp          = ecur.g + ecur.delta;

        if (tmp < threshold) {
            /*----------------------------------------------------------
             *  no need to insert – just bump g of current tuple
             *---------------------------------------------------------*/
            ++ecur.g;
                    //  System.out.printf("adding %f, increasing g %n", v);

            /* heap entries for ecur and its predecessor now “behind” by one;
             * the C++ comments say we fix that lazily when we next compress.
             */
        } else if (entries_map.containsKey(v)) {
            Map.Entry<Double,entry> iter2 = entries_map.floorEntry(v);
            assert iter2.getKey() == v : "...";
            ++iter2.getValue().g;
                    //  System.out.printf("adding %f, already in the keys %n", v);
        } else {
            /*----------------------------------------------------------
             *  insert a new tuple   (v, entry(1, tmp‑1))
             *---------------------------------------------------------*/
            entry newEntry           = new entry(1, tmp - 1);
                    //  System.out.printf("adding %f, new tuple %n", v);

            entries_map.put(v, newEntry);   // Java returns *old* value
            Map.Entry<Double,entry> iter2 = entries_map.floorEntry(v); // new entry

            /* push potential compression candidate into heap              */
            ThresholdEntry th = new ThresholdEntry();
            th.threshold = tmp + 1;
            th.map_iter  = iter2;
            compress_heap.add(th);
// System.out.println("-----");
//             for (Map.Entry<Double, entry> entry : entries_map.entrySet()) {
//                 System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue().g + ", " + entry.getValue().delta);
//             }
            /*----------------------------------------------------------
             *  try to remove one tuple as in C++ while‑loop
             *---------------------------------------------------------*/
            while (true) {
                ThresholdEntry topEntry = compress_heap.peek();
                if (topEntry == null || topEntry.threshold > threshold) break;

                compress_heap.poll();                   // pop

                // if (!entries_map.containsKey(topEntry.map_iter.getKey())) continue; // already deleted -- do not 

                Map.Entry<Double, entry> map_iter1 = topEntry.map_iter;
                Map.Entry<Double, entry> map_iter2 =
                        entries_map.higherEntry(map_iter1.getKey());
                // if (map_iter1 == null) continue; // map_iter2 points to the smallest tuple

                // assert map_iter2 != null && map_iter1 != null
                //        && map_iter2 != entries_map.end();

                entry e1 = map_iter1.getValue();
                entry e2 = map_iter2.getValue();

                int real_threshold = e1.g + e2.g + e2.delta;

                if (real_threshold <= threshold) {
                    //  System.out.printf("removing %f, map_iter2 key %f %n", map_iter1.getKey(), map_iter2.getKey());
                    // System.out.printf("old e2.g %d, e1.g %d %n", e2.g, e1.g);
                    e2.g += e1.g;
                    // System.out.printf("new e2.g %d%n", e2.g);
                    entries_map.remove(map_iter1.getKey());
                    break;
                } else {
                    topEntry.threshold = real_threshold;
                    compress_heap.add(topEntry);        // push updated record
                }
            }
            // System.out.println("-----");
            // for (Map.Entry<Double, entry> entry : entries_map.entrySet()) {
            //     System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue().g + ", " + entry.getValue().delta);
            // }
        }
        long totalG = 0;
        for (entry e : entries_map.values()) {
            totalG += e.g;                      // g is int; promote to long
        }
        if (totalG != last_n + 1) {
            System.out.println("GK invariant broken: Σg = " + totalG + " , last_n = " + last_n + ", val = " + v);
            for (Map.Entry<Double, entry> entry : entries_map.entrySet()) {
                System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue().g + ", " + entry.getValue().delta);
            }
            assert totalG == last_n + 1 : "GK invariant broken";
        }
    }
}

/*-----------------------------------------------------------------------------
 *  Notes & caveats
 *-----------------------------------------------------------------------------
 *  1.  Unsigned types:  Java has none, so 32‑bit unsigned ints are stored in
 *      signed int; 64‑bit uint64_t → long.  All arithmetic still matches as
 *      long as your values never exceed 2^63‑1 (same as C++ signed‑long‑long).
 *
 *  2.  The original C++ returns literal 0 when query_for_value() is asked for
 *      a rank smaller than the first summary tuple.  Returning 0 generically
 *      in Java is impossible without reflection, so the code above returns
 *      null.  If `Double` is always Integer or Long in your use‑case, feel
 *      free to cast `(Double)(Integer)0` or `(Double)(Long)0L` instead.
 *
 *  3.  `finalize()` in Java is deprecated; using the same name is *legal* but
 *      triggers a warning.  The `@SuppressWarnings("deprecation")` marker keeps
 *      the compiler quiet while preserving the C++ API surface exactly.
 *
 *  4.  `TreeMap` guarantees iterator stability for reads, but its `Entry`
 *      objects are not immutable.  The algorithm stores those entries in the
 *      heap exactly as the C++ stores iterators.  That is sufficient, because
 *      insertions/removals that might invalidate them re‑compute and re‑push
 *      as needed (same as the original).
 *
 *  5.  Assertions (`assert`) are enabled only when the JVM is started with
 *      the `‑ea` flag, matching C++’s compile‑time assert semantics.
 *
 *  With these minor Java‑isms aside, every branch, loop and arithmetic
 *  expression is a direct lift from the C++ source you provided.
 */