/*
 * Implementation of the GK algorithm
 * Copyright (c) 2013 Lu Wang <coolwanglu@gmail.com>
 * Translated to Java by ChatGPT and manually checked by Pavel Vesely <vesely@iuuk.mff.cuni.cz>
 */

/*
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
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
        // FOR TESTING:
        // long totalG = 0;
        // for (entry e : entries_map.values()) {
        //     totalG += e.g;                      // g is int; promote to long
        // }
        // if (totalG != last_n + 1) {
        //     System.out.println("GK invariant broken: Σg = " + totalG + " , last_n = " + last_n + ", val = " + v);
        //     for (Map.Entry<Double, entry> entry : entries_map.entrySet()) {
        //         System.out.println("Key: " + entry.getKey() + ". Value: " + entry.getValue().g + ", " + entry.getValue().delta);
        //     }
        //     assert totalG == last_n + 1 : "GK invariant broken";
        // }
    }

    /** Full, greedy compression pass using the standard GK condition.
     *  Removes tuples (except the sentinel) when  g_i + g_{i+1} + δ_{i+1} ≤ ⌊2·EPS·last_n⌋.
     *  This keeps memory small while preserving the GK error guarantee.
     */
    private void compressFull() {
        if (entries_map.size() <= 2) return; // only sentinel or one real + sentinel

        final int threshold = (int)Math.floor(EPS * last_n * 2);

        // Start at the first (smallest) entry and try to merge it into its successor.
        Map.Entry<Double, entry> prev = entries_map.firstEntry();
        if (prev == null) return;

        while (true) {
            Map.Entry<Double, entry> curr = entries_map.higherEntry(prev.getKey());
            if (curr == null) break;                 // nothing to the right
            if (curr.getKey().equals(max_v)) break;  // don't compress across the sentinel

            entry e1 = prev.getValue();
            entry e2 = curr.getValue();

            int realThreshold = e1.g + e2.g + e2.delta;

            if (realThreshold <= threshold) {
                // Merge prev into curr and re-check with the new predecessor of curr
                e2.g += e1.g;
                entries_map.remove(prev.getKey());

                Map.Entry<Double, entry> newPrev = entries_map.lowerEntry(curr.getKey());
                prev = (newPrev != null) ? newPrev : curr;  // if none, continue from curr
            } else {
                // Move window right
                prev = curr;
            }
        }
    }

    /**
     * Merge another GK sketch into this one.
     * Requirements: same EPS and same max_v (sentinel).
     * Complexity: O(|S1| + |S2|) plus a single linear-time compression pass.
     */
    public void merge(GK other) {
        if (other == null || other.last_n == 0) return;
        if (this.EPS != other.EPS) {
            throw new IllegalArgumentException("Cannot merge GK sketches with different EPS values.");
        }
        if (Double.compare(this.max_v, other.max_v) != 0) {
            throw new IllegalArgumentException("Cannot merge GK sketches with different sentinel (max_v) values.");
        }

        // Ensure summaries exist
        this.finalize();
        other.finalize();

        // Helper to iterate summaries without the sentinel at max_v
        List<SummaryEntry> s1 = new ArrayList<>(this.summary.size());
        for (SummaryEntry se : this.summary) if (!se.v.equals(max_v)) s1.add(se);
        List<SummaryEntry> s2 = new ArrayList<>(other.summary.size());
        for (SummaryEntry se : other.summary) if (!se.v.equals(max_v)) s2.add(se);

        // Merge two cumulative summaries (like merge-sort over v),
        // combining cumulative g's and taking max(delta) on ties.
        List<SummaryEntry> merged = new ArrayList<>(s1.size() + s2.size());
        int i = 0, j = 0;
        long g1 = 0, g2 = 0;

        while (i < s1.size() && j < s2.size()) {
            SummaryEntry a = s1.get(i);
            SummaryEntry b = s2.get(j);
            if (a.v.compareTo(b.v) < 0) {
                g1 = a.g;
                SummaryEntry m = new SummaryEntry();
                m.v = a.v;
                m.g = g1 + g2;
                m.delta = a.delta;
                merged.add(m);
                i++;
            } else if (a.v.compareTo(b.v) > 0) {
                g2 = b.g;
                SummaryEntry m = new SummaryEntry();
                m.v = b.v;
                m.g = g1 + g2;
                m.delta = b.delta;
                merged.add(m);
                j++;
            } else {
                // Same value in both summaries: combine
                g1 = a.g; g2 = b.g;
                SummaryEntry m = new SummaryEntry();
                m.v = a.v;
                m.g = g1 + g2;
                m.delta = Math.max(a.delta, b.delta);
                merged.add(m);
                i++; j++;
            }
        }
        while (i < s1.size()) {
            SummaryEntry a = s1.get(i++);
            g1 = a.g;
            SummaryEntry m = new SummaryEntry();
            m.v = a.v; m.g = g1 + g2; m.delta = a.delta;
            merged.add(m);
        }
        while (j < s2.size()) {
            SummaryEntry b = s2.get(j++);
            g2 = b.g;
            SummaryEntry m = new SummaryEntry();
            m.v = b.v; m.g = g1 + g2; m.delta = b.delta;
            merged.add(m);
        }

        // Rebuild entries_map from merged cumulative (g, delta),
        // and compute Σg over non-sentinel entries to set sentinel.g so Σg == last_n + 1.
        this.entries_map.clear();
        this.compress_heap.clear(); // not needed for compression pass we run now

        long prevG = 0;
        long sumGReal = 0;

        for (SummaryEntry s : merged) {
            int g = (int)(s.g - prevG);   // turn cumulative g back into local g
            int d = (int)s.delta;
            this.entries_map.put(s.v, new entry(g, d));
            sumGReal += g;
            prevG = s.g;
        }

        // Update total n and set sentinel so Σg == last_n + 1 (as in the original structure)
        this.last_n += other.last_n;

        long neededSentinelG = 1 + (this.last_n - sumGReal);
        if (neededSentinelG < 1) neededSentinelG = 1; // safety
        this.entries_map.put(this.max_v, new entry((int)neededSentinelG, 0));

        // Run a full greedy compression with the new global threshold
        compressFull();

        // Rebuild summary for queries
        this.finalize();
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