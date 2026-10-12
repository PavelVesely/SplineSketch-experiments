// based on https://github.com/coolwanglu/quantile-alg/blob/master/gk.h
// LICENSE (preserved)
/*
 * Implementation of the GK algorithm
 * Copyright (c) 2013 Lu Wang <coolwanglu@gmail.com>
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

// based on 
public class GK {

    public long   last_n;
    public double EPS;
    public double   max_v;

    /*-------------------------------- entry --------------------------------*/
    public static class entry {
        public int g;
        public int delta;

        public entry() { }
        public entry(int _g, int _d) { this.g = _g;  this.delta = _d; }
    }

    public final TreeMap<Double, entry> entries_map = new TreeMap<>();

    public int max_gd;

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
            return Integer.compare(this.threshold, o.threshold);
        }
    }
    public final PriorityQueue<ThresholdEntry> compress_heap =
            new PriorityQueue<>();

    /*------------------------------------------------------------------------
     *  Constructors
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