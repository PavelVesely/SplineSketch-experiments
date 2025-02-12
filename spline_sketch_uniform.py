import numpy as np
from scipy.interpolate import PchipInterpolator
import copy
import math

# note: assumes that lst is sorted
def equally_spaced_selection(lst, k):
    n = len(lst)
    if k > n:
        raise ValueError("k cannot be greater than the length of the list")
    if k <= 1:
        raise ValueError("k must be a positive integer >= 2")

    step = (n - 1) / (k - 1)
    thresholds = []

    for i in range(k):
        index = round(i * step)
        if index >= n:
            index = n - 1
        current = lst[index]
        thresholds.append(current)

    # somewhat technical handling of repeated values of thresholds
    i = 0
    while i < k:
        j = i+1
        while j < k and thresholds[i] >= thresholds[j] - abs(thresholds[j]) * 1e-9 - 1e-100:
            j += 1
        if j > i+1: # repeated values of thresholds
            prev = thresholds[i-1] if i > 0 else thresholds[0] - 1 # minus 1 is arbitrary
            next = thresholds[j] if j < k else thresholds[k-1] + 1 # plus 1 is arbitrary
            thresholds[i] = thresholds[i] - (thresholds[i] - prev) * 1e-9
            # note: thresholds[i+1] not modified
            for jj in range(i + 2, j):
                thresholds[jj] = thresholds[jj-1] + (next - thresholds[jj-1]) * 1e-3 / k
        i = j


    assert all(thresholds[i] < thresholds[i+1] for i in range(k-1)), thresholds

    result = []
    j = 0
    prevJ = 0
    for i in range(k):
        while j < n and lst[j] <= thresholds[i]:
            j += 1
        result.append([thresholds[i], j - prevJ, False])
        prevJ = j
    return result

def merge_array_into_buckets(array, buckets):
    # we assume that the array is sorted
    # Note: does not add all array elements into the buckets when the buckets do not cover all of the array elements, specifically when max(array) > max(bucket bound).
    #     This only happens when called for the first time in the consolidate method, so it's not a bug.
    #     Also note the asymmetry here: new minimum is added into the first bucket. Perhaps we should not add elements smaller than buckets[0][0] into any bucket

    idx_buckets = 0
    idx_array = 0

    while idx_array < len(array) and idx_buckets < len(buckets):
        while idx_array < len(array) and buckets[idx_buckets][0] >= array[idx_array]:
            buckets[idx_buckets][1] += 1
            idx_array += 1
        idx_buckets += 1


class SplineSketchUniform:
    def __init__(self, k, print_info=""):
        assert k >= 6
        self.k = k
        self.buckets = [] # list of triples: [bucket_threshold (on the right), bucket counter, is protected?]
        self.buffer = []
        self.n = 0

        self.print_info = print_info # for debugging

        self.split_join_ratio = 1.5 # for heuristic error (const.)
        self.buffer_size_bound = 5*k # (const.)

        self.min_relative_bucket_length = 1e-8 # default Python precision is about 15, but we use less (const.)
        self.min_absolute_nonzero_value = float('inf') # SHOULD TO BE STORED in updatable mode only

        self.min_frac_bucket_bound_to_split = 0.01 # what fraction of bound a bucket needs to exceed so that it gets split due to large heur. error (const.)

        self.epoch_incr_factor = 1.25
        self.epoch_end = 2 * self.buffer_size_bound 

        self.default_bucket_bound_mult = 3 # constant
        self.bucket_bound_mult = 3 # may increase over time, reset to default_bucket_bound_mult at epoch end, but does not need to be stored when the sketch is serialized

        # for stats on num. iterations
        self.total_iter_cnt = 0
        self.num_consolidates = 0

    def print_avg_iters_stats(self): # for testing
        print(f"after {self.n} updates, avg. num. iterations during consolidating is {self.total_iter_cnt / self.num_consolidates} ({self.total_iter_cnt} / {self.num_consolidates})")

    
    # num. of bytes to store the sketch in an updatable form (with protected array, but assuming buffer is merged into buckets)
    # a threshold takes 8 bytes, a counter 8 bytes
    # + 8 bytes for min_absolute_nonzero_value
    # (note: self.n == sum of couters of buckets, so no need to store n)
    def serializedSketchBytesUpdatable(self):
        return self.k * 16 + int(math.ceil(self.k / 8.0)) + 8

    # num. of bytes to store the sketch in a compact form, i.e., just bucket thresholds and counters
    def serializedSketchBytesCompact(self):
        return self.k * 16

    def are_sufficiently_different(self, a, b): # returns True when a and b are sufficiently different (in both the relative and absolute sense)
        return abs(a-b) / max(abs(a), abs(b), self.min_absolute_nonzero_value) > self.min_relative_bucket_length

    def consolidate(self):
        if len(self.buffer) == 0 and len(self.buckets) == self.k: # sketch has clean buffer and the right number of buckets
            return

        if self.buckets == []:
            if len(self.buffer) < self.k: # too few items to create k buckets
                return
            self.buffer.sort()
            self.buckets = equally_spaced_selection(self.buffer, self.k)
            self.buffer = []
            return
        
        # make buckets non-protected after epoch end
        if self.n >= self.epoch_end:
            for i in range(len(self.buckets)):
                self.buckets[i][2] = False
            # next epoch end -- could be less than self.n + self.buffer_size_bound, but this is not an issue (at the beginning, there's an epoch end during every consolidate)
            self.epoch_end = int(self.epoch_incr_factor * self.n)
            self.bucket_bound_mult = self.default_bucket_bound_mult

        # we need to process the buffer and we already have buckets

        self.buffer.sort()

        assert abs(sum([count for _, count, _ in self.buckets]) + len(self.buffer) - self.n) < 1e-3, (sum([count for _, count, _ in self.buckets]), self.n)

        # create a copy of the buckets
        buckets_new = copy.deepcopy(self.buckets)
        merge_array_into_buckets(self.buffer, buckets_new)
        
        # calculate spline for the new buckets
        assert all(math.isfinite(b[0]) for b in buckets_new), f"a bucket with a non finite boundary exists: {buckets_new}"
        assert all(math.isfinite(b[1]) for b in buckets_new), f"a bucket with a non finite counter exists: {buckets_new}"

        iter = 0
        performed_changes = True
        while performed_changes:
            performed_changes = False

            # get indexes of buckets that have to be split,
            # the 1.01 is to deal with rounding errors;
            # the second condition is to ensure that bucket length is relatively not too small to be split, and also the bucket boundaries are not too close to zero
            have_to_split = {i for i in range(1,len(buckets_new)) if buckets_new[i][1] > 1.01*self.bucket_bound() and \
                            self.are_sufficiently_different(buckets_new[i][0], buckets_new[i-1][0]) } 
            
            # get indexes of buckets that can be joined, i.e., will be at most 75% full after the join and we do not remove a protected threshold
            can_be_joined = [i for i in range(1,len(buckets_new)-1) if
                             buckets_new[i][1] + buckets_new[i+1][1] <= 0.75*self.bucket_bound() \
                             and not buckets_new[i][2]]
            can_be_joined.sort(key=lambda x: self.error_estimate(x, join=True, buckets=buckets_new))

            # filter out buckets that are relatively or absolutely too small to be split or have counters relatively small (1% of the bound), and sort by heuristic error
            buckets_by_errors = sorted(list(i for i in range(len(buckets_new)) \
                                            if self.are_sufficiently_different(buckets_new[i][0], buckets_new[i-1][0]) and \
                                                buckets_new[i][1] > self.min_frac_bucket_bound_to_split * self.bucket_bound()), \
                                            key=lambda x: self.error_estimate(x, buckets=buckets_new), reverse=True)

            # did we get extremal points? if so, we set new_extremes to 1 or 2 depending whethwe we only got new min/max or both
            new_extremes = 0
            if len(self.buffer) > 0:
                if min(self.buffer) < buckets_new[0][0]:
                    new_extremes += 1
                if max(self.buffer) > buckets_new[-1][0]:
                    new_extremes += 1

            resize_diff = len(buckets_new) - self.k # used when merging a sketch into self or resizing the sketch -- the number of buckets may not be k when we call consolidate
    
            will_be_split = copy.deepcopy(have_to_split)
            will_be_joined = set()
                
            idx_join = 0

            # we add the buckets that will be joined when splitting stuff that has to be split
            # we also add the buckets that will be joined because of adding new extremal buckets and handle resizing the sketch, i.e., changing the number of buckets
            while len(will_be_joined) < len(will_be_split) + new_extremes + resize_diff and idx_join < len(can_be_joined):

                if can_be_joined[idx_join]-1 not in will_be_joined \
                    and can_be_joined[idx_join]+1 not in will_be_joined:

                    assert can_be_joined[idx_join] not in will_be_split # o/w idx_join would not be in can_be_joined
                    assert can_be_joined[idx_join]+1 not in will_be_split # o/w idx_join would not be in can_be_joined
                    assert can_be_joined[idx_join] not in will_be_joined
                    will_be_joined.add(can_be_joined[idx_join])
                idx_join += 1

            if len(will_be_joined) == 0 and len(will_be_split) + new_extremes + resize_diff > 0: # we ran out of buckets to join and there will be no progress in this iteration
                assert resize_diff != 0 or not all(buckets_new[i][2] == False for i in range(len(buckets_new))) # the latter could happen during merging sketches (but probably not an issue)
                assert idx_join >= len(can_be_joined)
                self.bucket_bound_mult *= 2 # increase bucket_bound_mult twice
                print(f"!! idx_join >= len(can_be_joined) => increasing bucket bound multiplier to {self.bucket_bound_mult} at n={self.n}, resize diff={resize_diff}, epoch end = {self.epoch_end}, iter {iter}, k = {self.k} (info: {self.print_info})")
                #print(will_be_split, can_be_joined, will_be_joined, new_extremes, resize_diff)
                performed_changes = True
                iter += 1
                continue # run next iteration

            idx_split = 0

            # when expanding the sketch size (increasing k), some buckets may be split without any join
            while idx_split < -(len(will_be_split) + new_extremes + resize_diff) and idx_split < len(buckets_by_errors):
                # skip the buckets that are already to be split
                if buckets_by_errors[idx_split] in have_to_split \
                    or buckets_by_errors[idx_split] == 0 \
                    or buckets_by_errors[idx_split] in will_be_joined \
                    or buckets_by_errors[idx_split]-1 in will_be_joined:
                    idx_split += 1
                    continue
                will_be_split.add(buckets_by_errors[idx_split])
                idx_split += 1
                

            # we also split some more buckets if it is beneficial w.r.t. the error estimate
            # Update: we now also leave a constant fraction of the buckets that can be joined --- this is to ensure we have enough buckets to join to the "have_to_split" buckets during any epoch;
            # the -2 is there because of new extremes
            while idx_join < len(can_be_joined) - self.k/4 - 2 and idx_split < len(buckets_by_errors) and \
                self.error_estimate(buckets_by_errors[idx_split], buckets=buckets_new) > \
                self.split_join_ratio * self.error_estimate(can_be_joined[idx_join], join=True, buckets=buckets_new):
                # skip the buckets that are already to be split
                if buckets_by_errors[idx_split] in have_to_split \
                    or buckets_by_errors[idx_split] == 0 \
                    or buckets_by_errors[idx_split] in will_be_joined \
                    or buckets_by_errors[idx_split]-1 in will_be_joined:
                    idx_split += 1
                    continue

                # skip the buckets we cannot join because a neighbor is already in will_be_joined
                if can_be_joined[idx_join]-1 in will_be_joined \
                    or can_be_joined[idx_join]+1 in will_be_joined \
                    or can_be_joined[idx_join] in will_be_split \
                    or can_be_joined[idx_join]+1 in will_be_split:
                    idx_join += 1
                    continue

                if buckets_by_errors[idx_split] == can_be_joined[idx_join] or buckets_by_errors[idx_split] == can_be_joined[idx_join] + 1:
                    idx_join += 1
                    continue

                will_be_split.add(buckets_by_errors[idx_split])
                will_be_joined.add(can_be_joined[idx_join])
                idx_split += 1
                idx_join += 1

            #assert max(len(will_be_split) + new_extremes + resize_diff, 0) == len(will_be_joined), f"{will_be_split} {will_be_joined}"
            assert 0 not in will_be_split
            assert 0 not in will_be_joined
            assert len(buckets_new)-1 not in will_be_joined

            if len(will_be_split) > 0 or len(will_be_joined) > 0: # due to resizing (e.g. when joining), these two conditions are not equivalent
                performed_changes = True

            # we change the bucket boundaries
            new_bucket_boundaries = []
            new_protected = []
            previous_split = False
            for i in range(len(buckets_new)):
                if i in will_be_joined:
                    assert i not in will_be_split, (i, buckets_new[i][1], self.n, self.k, sum([count for _, count in buckets_new]))
                    assert i+1 not in will_be_split
                    assert i+1 not in will_be_joined
                    assert i-1 not in will_be_joined

                    continue # skip the current right bucket boundary, effectively joining buckets i-1 and i

                if i in will_be_split:
                    assert i not in will_be_joined
                    assert i-1 not in will_be_joined
                    assert i != 0

                    new_bucket_boundaries.append((buckets_new[i][0] + buckets_new[i-1][0]) / 2) # always split in the middle, it's an approx. median by properties of PChipInterpolator
                    
                    if len(new_protected) > 0:
                        new_protected[-1] = True
                    new_protected.append(True)
                    previous_split = True

                new_bucket_boundaries.append(buckets_new[i][0])
                new_protected.append(buckets_new[i][2] | previous_split)
                previous_split = False
            
            if len(self.buffer) > 0:
                if min(self.buffer) < new_bucket_boundaries[0]:
                    new_bucket_boundaries = [min(self.buffer)] + new_bucket_boundaries
                    new_protected = [False] + new_protected
                    performed_changes = True
                if max(self.buffer) > new_bucket_boundaries[-1]:
                    new_bucket_boundaries.append(max(self.buffer))
                    new_protected.append(False)
                    performed_changes = True

            assert len(new_bucket_boundaries) == len(new_protected)
            #assert len(new_bucket_boundaries) == self.k, (len(new_bucket_boundaries), self.k) # may not hold now
            assert all(new_bucket_boundaries[i] < new_bucket_boundaries[i+1] for i in range(len(new_bucket_boundaries) - 1))
            assert len(self.buffer) == 0 or new_bucket_boundaries[0] <= min(self.buffer)
            assert len(self.buffer) == 0 or new_bucket_boundaries[-1] >= max(self.buffer)
            assert new_bucket_boundaries[0] <= self.buckets[0][0]
            assert new_bucket_boundaries[-1] >= self.buckets[-1][0]

            if performed_changes:
                interpolator = self.calc_spline(self.buckets)
                buckets_new = [[new_bucket_boundaries[0], interpolator(new_bucket_boundaries[0]), False]]
                for i in range(1, len(new_bucket_boundaries)):
                    buckets_new.append([new_bucket_boundaries[i], \
                                        interpolator(new_bucket_boundaries[i]) - interpolator(new_bucket_boundaries[i-1]), \
                                        new_protected[i]])

                merge_array_into_buckets(self.buffer, buckets_new)
            
            #assert len(buckets_new) == self.k  # may not hold now
            
            iter += 1


        # we have processed the buffer and we have the new buckets
        self.buckets = buckets_new
        assert len(self.buckets) == self.k

        assert abs(sum([count for _, count, _ in self.buckets]) - self.n) < 1e-1, (sum([count for _, count, _ in self.buckets]), self.n)

        # do not forget to clear the buffer
        self.buffer = []

        # stats on num. iters
        self.total_iter_cnt += iter
        self.num_consolidates += 1

    def update(self, item):
        if math.isinf(item) or math.isnan(item):
            print(f"cannot add {item}")
            return
        self.buffer.append(item)
        self.n += 1
        if abs(item) > 0 and abs(item) < self.min_absolute_nonzero_value:
            self.min_absolute_nonzero_value = abs(item)

        if len(self.buffer) < self.buffer_size_bound:
            return

        self.consolidate()
        
    def resize(self, new_k):
        assert new_k >= 6
        if new_k != self.k:
            if abs(self.k - new_k) > 0.25 * self.k: # substantial relative difference => force new epoch
                self.epoch_end = 0
            self.k = new_k
            self.buffer_size_bound = 5*self.k
            self.consolidate()

    def merge(sketch1, sketch2):
        if sketch1.n >= sketch2.n: # merge into the larger sketch
            sketch1.mergeIntoSelf(sketch2)
            return sketch1
        else:
            sketch2.mergeIntoSelf(sketch1)
            return sketch2

    # notes: params. inhereted from self, including the epoch end (which is not changed). So if two similarly sized sketches are merged, the opoch ends, but also the bucket bound increases substantially
    def mergeIntoSelf(self, other):
        spl = self.calc_spline() # must be done before adjusting self.n
        splOther = other.calc_spline()

        self.min_absolute_nonzero_value = min(self.min_absolute_nonzero_value, other.min_absolute_nonzero_value)
        self.n += other.n
        self.buffer.extend(other.buffer)
        self.total_iter_cnt += other.total_iter_cnt
        self.num_consolidates += other.num_consolidates
        # create a union of buckets (union bucket boundaries, up to very similar ones; counters calculated using splines; protection inhereted from self and reset otherwise)
        buckets_union = []
        i = 0
        j = 0
        prevSum = 0
        assert len(self.buckets) == 0 or len(self.buckets) == self.k, (self.k, len(self.buckets))
        assert len(other.buckets) == 0 or len(other.buckets) == other.k, (other.k, len(other.buckets))
        if len(other.buckets) == 0:
            buckets_union = self.buckets
        elif len(self.buckets) == 0:
            buckets_union = other.buckets
        else:
            while i < self.k or j < other.k:
                if i < self.k and (j >= other.k or self.buckets[i][0] <= other.buckets[j][0]):
                    t = self.buckets[i][0]
                    newSum = spl(t) + splOther(t)
                    buckets_union.append([t, newSum - prevSum, self.buckets[i][2]]) # protection inhereted here (may later get reset due to epoch end)
                    prevSum = newSum
                    i += 1
                else:
                    t = other.buckets[j][0]
                    newSum = spl(t) + splOther(t)
                    buckets_union.append([t, newSum - prevSum, False]) # protection reset in the other sketch
                    prevSum = newSum
                    j += 1
            buckets_union = self.prune_too_close_thresholds(buckets_union)
            if len(buckets_union) > 0 and buckets_union[-1][0] < max(self.buckets[-1][0], other.buckets[-1][0]): # could happen when skipping a threshold
                print(f"missing max: {buckets_union[-1][0]}, {self.buckets[-1][0]}, {other.buckets[-1][0]}")
                # buckets_union[-1][0] = max(self.buckets[-1][0], other.buckets[-1][0])
                # assert self.n - len(self.buffer) - sum([cnt for _, cnt, _ in buckets_union]) >= 0, f"missing max: {buckets_union[-1][0]}, {self.buckets[-1][0]}, {other.buckets[-1][0]}, n = {self.n}, {sum([cnt for _, cnt, _ in buckets_union])} "
                # buckets_union[-1][1] += self.n - len(self.buffer) - sum([cnt for _, cnt, _ in buckets_union])

        self.buckets = buckets_union
        if (len(self.buckets) != self.k and len(self.buckets) != 0) or len(self.buffer) > self.buffer_size_bound:
            self.consolidate()
            
    def prune_too_close_thresholds(self, buckets):
        if buckets == None:
            buckets = self.buckets
        buckets_new = [buckets[0]]
        lastT = buckets[0][0]
        cnt = 0
        prot = False
        for i in range(1, len(buckets) - 1):
            if self.are_sufficiently_different(buckets[i][0], lastT):
                buckets_new.append([buckets[i][0], cnt + buckets[i][1], prot | buckets[i][2]])
                lastT = buckets[i][0]
                cnt = 0
                prot = False
            else:
                cnt += buckets[i][1]
                prot |= buckets[i][2]
        i = len(buckets) - 1
        if not self.are_sufficiently_different(buckets[i][0], lastT) and len(buckets) > 1:
            cnt += buckets_new[-1][1]
            prot |= buckets_new[-1][2]
            buckets_new.pop(-1)
        buckets_new.append([buckets[i][0], cnt + buckets[i][1], prot | buckets[i][2]])
        return buckets_new

    # returns pchip spline interpolation for the buckets (not including the buffer)
    def calc_spline(self, buckets=None):
        if buckets is None:
            buckets = self.buckets
        if len(buckets) == 0:
            def zero_fnct(x):
                return 0
            return zero_fnct
        prefix_sums = np.cumsum([count for _, count, _ in buckets])
        spl = PchipInterpolator([x[0] for x in buckets], prefix_sums, extrapolate=False)
        def spl_fixed(x: float) -> int: # for x smaller than min or larger than max, nan is returned in spl
            if x < buckets[0][0]: 
                return 0
            elif x >= buckets[-1][0]:
                return prefix_sums[-1]
            else:
                return int(spl(x))
        return spl_fixed

    def query(self, items):
        np_buffer = np.sort(np.array(self.buffer))
        ranks_within_buffer = np.searchsorted(np_buffer, items, side='right')

        if len(self.buckets) > 0:
            spline = self.calc_spline()
            ranks_within_sketch = np.array(list(map(spline, items)))
            return ranks_within_buffer + ranks_within_sketch
        else:
            return ranks_within_buffer
        
    def error_estimate(self, idx_bucket, join=False, buckets=None):
        if buckets is None:
            buckets = self.buckets

        # the point of this function is that it can handle indexes outside buckets array bounds when it returns a default of 0
        def get_bucket_cnt(i):
            if i < 0 or i >= len(buckets):
                return 0
            return buckets[i][1]

        # the point of this function is that it can handle indexes outside buckets array bounds when it returns a default of infty
        def get_bucket_length(i):
            if i <= 0:
                return buckets[1][0] - buckets[0][0] # using the leftmost bucket length as default
            if i >= len(buckets):
                return buckets[len(buckets)-1][0] - buckets[len(buckets)-2][0] # using the rightmost bucket length as default
            return buckets[i][0] - buckets[i - 1][0]
        
        # 2nd derivative of CDF = 1st derivative of pdf
        def get_der2(bucket_length, cnt, prevLength, prevCount, nextLength, nextCount):
            return max(abs(nextCount/nextLength - cnt/bucket_length) / (bucket_length + nextLength), \
                       abs(cnt/bucket_length - prevCount/prevLength) / (bucket_length + prevLength))

        if not join:
            bucket_length = buckets[idx_bucket][0] - buckets[idx_bucket-1][0] # get_bucket_length(buckets, idx_bucket)
            cnt = buckets[idx_bucket][1] #get_bucket_cnt(buckets, idx_bucket)
            prevLength = get_bucket_length(idx_bucket-1)
            prevCount = get_bucket_cnt(idx_bucket-1)
            nextLength = get_bucket_length(idx_bucket+1)
            nextCount = get_bucket_cnt(idx_bucket+1)
            der2 = get_der2(bucket_length, cnt, prevLength, prevCount, nextLength, nextCount)
        else:
            bucket_length = buckets[idx_bucket+1][0] - buckets[idx_bucket-1][0]
            cnt = buckets[idx_bucket][1] + buckets[idx_bucket+1][1]
            prevLength = get_bucket_length(idx_bucket-1)
            prevCount = get_bucket_cnt(idx_bucket-1)
            nextLength = get_bucket_length(idx_bucket+2)
            nextCount = get_bucket_cnt(idx_bucket+2)
            der2 = get_der2(bucket_length, cnt, prevLength, prevCount, nextLength, nextCount)
        
        return bucket_length**2 * abs(der2)

    def bucket_bound(self):
        return self.bucket_bound_mult*self.n/self.k # self.bucket_bound_mult gets increased if needed
        

