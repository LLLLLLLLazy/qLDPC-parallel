# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True
import numpy as np
cimport numpy as cnp


def osd_list_candidates(
    cnp.ndarray[cnp.uint8_t, ndim=2] mat,
    cnp.ndarray[cnp.uint8_t, ndim=1] syndrome,
    cnp.ndarray[cnp.float64_t, ndim=1] prior,
    cnp.ndarray[cnp.uint8_t, ndim=1] seed,
    cnp.ndarray[cnp.float64_t, ndim=1] reliabilities,
    int top_k,
):
    cdef int rows = mat.shape[0]
    cdef int cols = mat.shape[1]
    cdef int row = 0
    cdef int col, r, c, pivot, other, i, j, k
    cdef int pivot_count = 0
    cdef unsigned char tmp, parity
    cdef double p, cost

    if top_k <= 1:
        out = np.zeros((1, cols), dtype=np.uint8)
        out[0, :] = seed
        return out

    order = np.argsort(-np.asarray(reliabilities, dtype=np.float64))
    h = np.ascontiguousarray(mat[:, order], dtype=np.uint8)
    rhs = np.array(syndrome, dtype=np.uint8, copy=True, order="C")
    seed_perm = np.ascontiguousarray(seed[order], dtype=np.uint8)
    prior_perm = np.ascontiguousarray(prior[order], dtype=np.float64)
    reliability_perm = np.ascontiguousarray(reliabilities[order], dtype=np.float64)

    cdef cnp.uint8_t[:, ::1] H = h
    cdef cnp.uint8_t[::1] RHS = rhs
    cdef cnp.uint8_t[::1] Seed = seed_perm
    cdef cnp.float64_t[::1] Prior = prior_perm
    cdef cnp.float64_t[::1] Reliability = reliability_perm

    pivot_cols = np.empty(min(rows, cols), dtype=np.int32)
    pivot_rows = np.empty(min(rows, cols), dtype=np.int32)
    cdef cnp.int32_t[::1] PivotCols = pivot_cols
    cdef cnp.int32_t[::1] PivotRows = pivot_rows

    for col in range(cols):
        pivot = -1
        for r in range(row, rows):
            if H[r, col] != 0:
                pivot = r
                break
        if pivot < 0:
            continue
        if pivot != row:
            for c in range(cols):
                tmp = H[row, c]
                H[row, c] = H[pivot, c]
                H[pivot, c] = tmp
            tmp = RHS[row]
            RHS[row] = RHS[pivot]
            RHS[pivot] = tmp

        for other in range(rows):
            if other != row and H[other, col] != 0:
                for c in range(col, cols):
                    H[other, c] ^= H[row, c]
                RHS[other] ^= RHS[row]

        PivotCols[pivot_count] = col
        PivotRows[pivot_count] = row
        pivot_count += 1
        row += 1
        if row == rows:
            break

    for r in range(row, rows):
        if RHS[r] == 0:
            continue
        pivot = 0
        for c in range(cols):
            if H[r, c] != 0:
                pivot = 1
                break
        if pivot == 0:
            out = np.zeros((1, cols), dtype=np.uint8)
            out[0, :] = seed
            return out

    pivot_mask = np.zeros(cols, dtype=np.uint8)
    cdef cnp.uint8_t[::1] PivotMask = pivot_mask
    for i in range(pivot_count):
        PivotMask[PivotCols[i]] = 1

    free_cols_list = []
    for col in range(cols):
        if PivotMask[col] == 0:
            free_cols_list.append(col)
    free_cols = np.asarray(free_cols_list, dtype=np.int32)
    cdef int free_count = free_cols.shape[0]
    cdef cnp.int32_t[::1] FreeCols = free_cols

    if free_count == 0:
        candidates = np.zeros((1, cols), dtype=np.uint8)
        for i in range(pivot_count):
            candidates[0, PivotCols[i]] = RHS[PivotRows[i]]
        restored = np.zeros((1, cols), dtype=np.uint8)
        restored[0, order] = candidates[0]
        return restored

    free_reliability = np.empty(free_count, dtype=np.float64)
    cdef cnp.float64_t[::1] FreeReliability = free_reliability
    for i in range(free_count):
        FreeReliability[i] = Reliability[FreeCols[i]]
    low_free_order = np.argsort(free_reliability)

    search_free_count = min(free_count, max(4, top_k + 2))
    flip_sets = [()]
    for i in range(search_free_count):
        flip_sets.append((int(low_free_order[i]),))
    for i in range(search_free_count):
        for j in range(i + 1, search_free_count):
            flip_sets.append((int(low_free_order[i]), int(low_free_order[j])))
            if len(flip_sets) >= max(4 * top_k, top_k + 1):
                break
        if len(flip_sets) >= max(4 * top_k, top_k + 1):
            break

    raw_candidates = np.zeros((len(flip_sets), cols), dtype=np.uint8)
    raw_costs = np.full(len(flip_sets), np.inf, dtype=np.float64)
    cdef cnp.uint8_t[:, ::1] Raw = raw_candidates
    cdef cnp.float64_t[::1] Costs = raw_costs

    free_values = np.empty(free_count, dtype=np.uint8)
    cdef cnp.uint8_t[::1] FreeValues = free_values

    for k, flips in enumerate(flip_sets):
        for i in range(cols):
            Raw[k, i] = 0
        for i in range(free_count):
            FreeValues[i] = Seed[FreeCols[i]]
        for flip in flips:
            FreeValues[flip] ^= 1
        for i in range(free_count):
            Raw[k, FreeCols[i]] = FreeValues[i]

        for i in range(pivot_count - 1, -1, -1):
            parity = 0
            r = PivotRows[i]
            col = PivotCols[i]
            for j in range(free_count):
                if H[r, FreeCols[j]] != 0 and FreeValues[j] != 0:
                    parity ^= 1
            Raw[k, col] = RHS[r] ^ parity

        cost = 0.0
        for i in range(cols):
            if Raw[k, i] != 0:
                p = Prior[i]
                if p < 1e-12:
                    p = 1e-12
                elif p > 1 - 1e-12:
                    p = 1 - 1e-12
                cost += np.log((1.0 - p) / p)
        Costs[k] = cost

    sorted_idx = np.argsort(raw_costs)
    out_perm = np.zeros((top_k, cols), dtype=np.uint8)
    cdef cnp.uint8_t[:, ::1] OutPerm = out_perm
    cdef int out_count = 0
    cdef bint duplicate
    for idx in sorted_idx:
        if not np.isfinite(raw_costs[idx]):
            continue
        duplicate = False
        for i in range(out_count):
            duplicate = True
            for j in range(cols):
                if OutPerm[i, j] != Raw[idx, j]:
                    duplicate = False
                    break
            if duplicate:
                break
        if duplicate:
            continue
        for j in range(cols):
            OutPerm[out_count, j] = Raw[idx, j]
        out_count += 1
        if out_count == top_k:
            break

    if out_count == 0:
        out_perm[0, :] = seed_perm
        out_count = 1
    while out_count < top_k:
        out_perm[out_count, :] = out_perm[out_count - 1, :]
        out_count += 1

    out = np.zeros((top_k, cols), dtype=np.uint8)
    out[:, order] = out_perm
    return out
