# Mathematical and Correctness Analysis

## Executive Summary

This document identifies mathematical errors, edge cases, and correctness issues found in the HubnessDetector codebase. Several critical issues were identified that could lead to runtime errors or incorrect results.

---

## Critical Issues

### 1. **Stability Detector - Mathematical Error in Score Calculation**

**Location**: `hubscan/core/detectors/stability.py:135-138`

**Issue**: The stability score calculation is mathematically incorrect. The current implementation divides the retrieval count by `total_perturbations`, but this doesn't properly normalize the score.

**Current Code**:
```python
stability_scores[doc_idx] = count / total_perturbations
```

**Problem**: 
- A document can appear multiple times in the top-k results for a single query perturbation
- The maximum possible count is `num_queries * perturbations * k`, not `total_perturbations`
- The score doesn't account for the fact that a document appearing in position 1 vs position k should be weighted differently
- The normalization doesn't consider that a document appearing in all k positions for all perturbations would have `count = num_queries * perturbations * k`, giving a score > 1.0

**Impact**: Stability scores are incorrectly normalized and may exceed 1.0, making them incomparable and misleading.

**Recommendation**: 
1. Normalize by the maximum possible count: `max_count = num_queries * perturbations * k`
2. Or normalize per query: `stability_per_query = count / (perturbations * k)` then average
3. Consider position weighting (e.g., inverse rank weighting)

---

### 2. **Stability Detector - Edge Case: Empty Queries**

**Location**: `hubscan/core/detectors/stability.py:110`

**Issue**: `np.random.choice()` will fail if `len(queries) == 0`

**Current Code**:
```python
num_queries_to_use = min(len(queries), 1000)
query_indices = np.random.choice(len(queries), num_queries_to_use, replace=False)
```

**Problem**: If `queries` is empty, `len(queries) == 0`, and `np.random.choice(0, 0, replace=False)` will raise a `ValueError`.

**Impact**: Runtime crash when no queries are available.

**Recommendation**: Add check:
```python
if len(queries) == 0:
    logger.warning("No queries available for stability detection")
    return DetectorResult(scores=np.zeros(len(doc_embeddings)))
```

---

### 3. **Hubness Detector - Edge Case: Division by Zero**

**Location**: `hubscan/core/detectors/hubness.py:117`

**Issue**: Division by M (number of queries) without checking if M > 0

**Current Code**:
```python
hub_rate = hits.astype(np.float32) / M
```

**Problem**: If `M == 0` (no queries), this will cause a division by zero error.

**Impact**: Runtime crash when no queries are provided.

**Recommendation**: Add check:
```python
if M == 0:
    logger.warning("No queries provided for hubness detection")
    return DetectorResult(scores=np.zeros(len(doc_embeddings)))
```

---

### 4. **Hubness Detector - Validation Inconsistency**

**Location**: `hubscan/core/detectors/hubness.py:185`

**Issue**: Exact validation uses a subset of queries but compares against hub_rate calculated from all queries.

**Current Code**:
```python
exact_hub_rate = exact_hits.astype(np.float32) / num_validation
```

**Problem**: 
- `approx_hub_rate` is calculated using all M queries
- `exact_hub_rate` is calculated using only `num_validation` queries
- The comparison is not fair - they use different denominators

**Impact**: Validation metrics (overlap, correlation) are misleading because they compare hub rates calculated from different numbers of queries.

**Recommendation**: Either:
1. Calculate `approx_hub_rate` using only the same validation queries, or
2. Scale `exact_hub_rate` to match the full query set: `exact_hub_rate * (M / num_validation)`

---

### 5. **Query Sampling - Edge Case: Empty Document Set**

**Location**: `hubscan/core/sampling/queries.py:90`

**Issue**: `np.random.choice()` will fail if `n == 0`

**Current Code**:
```python
indices = self.rng.choice(n, min(num_queries, n), replace=False)
```

**Problem**: If `n == 0` (no documents), `np.random.choice(0, 0, replace=False)` will raise a `ValueError`.

**Impact**: Runtime crash when trying to sample queries from an empty document set.

**Recommendation**: Add check:
```python
if n == 0:
    raise ValueError("Cannot sample queries from empty document set")
```

---

### 6. **Cluster Spread Detector - Edge Case: Empty Queries**

**Location**: `hubscan/core/detectors/cluster_spread.py:110`

**Issue**: Batch processing will fail if `M == 0`

**Current Code**:
```python
for i in range(0, M, batch_size):
    end = min(i + batch_size, M)
    batch_queries = queries[i:end]
```

**Problem**: If `M == 0`, the loop won't execute, but subsequent code may assume queries were processed.

**Impact**: May return incorrect results (all zeros) without warning.

**Recommendation**: Add explicit check:
```python
if M == 0:
    logger.warning("No queries provided for cluster spread detection")
    return DetectorResult(scores=np.zeros(len(doc_embeddings)))
```

---

## Moderate Issues

### 7. **Stability Detector - Non-Reproducible Randomness**

**Location**: `hubscan/core/detectors/stability.py:118`

**Issue**: Uses `np.random.normal()` without a seed, making results non-reproducible.

**Current Code**:
```python
perturbed = query + np.random.normal(0, self.sigma, query.shape).astype(np.float32)
```

**Problem**: Even if the main random seed is set, this uses the global numpy random state, which may have been affected by other operations.

**Impact**: Results are not reproducible, making debugging and validation difficult.

**Recommendation**: Use a seeded random number generator:
```python
rng = np.random.default_rng(self.config.scan.seed if hasattr(self, 'config') else 42)
perturbed = query + rng.normal(0, self.sigma, query.shape).astype(np.float32)
```

---

### 8. **Dedup Detector - Potential Index Error**

**Location**: `hubscan/core/detectors/dedup.py:119`

**Issue**: Uses `distances[i]` without checking if `i < len(distances)`

**Current Code**:
```python
close_neighbors = np.sum(distances[i] < self.duplicate_threshold)
```

**Problem**: If `k_search > sample_size`, the distances array may have fewer rows than expected, though this is unlikely given the `min(10, sample_size)` check.

**Impact**: Potential IndexError in edge cases.

**Recommendation**: Add bounds checking or ensure `k_search <= sample_size`.

---

### 9. **Cluster Spread Detector - Entropy Normalization Edge Case**

**Location**: `hubscan/core/detectors/cluster_spread.py:140-144`

**Issue**: When `num_clusters_actual == 1`, `max_entropy = 1.0` but entropy will always be 0.

**Current Code**:
```python
max_entropy = np.log(num_clusters_actual) if num_clusters_actual > 1 else 1.0
if max_entropy > 0:
    normalized_entropy = cluster_entropy / max_entropy
```

**Problem**: If only one cluster exists, entropy is always 0 (no spread), but dividing by 1.0 is correct. However, this case should be handled explicitly.

**Impact**: Low - the result is correct but the logic could be clearer.

**Recommendation**: Add explicit handling:
```python
if num_clusters_actual <= 1:
    # No spread possible with single cluster
    normalized_entropy = np.zeros(N)
else:
    max_entropy = np.log(num_clusters_actual)
    normalized_entropy = cluster_entropy / max_entropy
```

---

## Minor Issues / Code Quality

### 10. **Robust Z-Score - Epsilon Handling**

**Location**: `hubscan/utils/metrics.py:36-37`

**Issue**: Uses `np.finfo(float).eps` when MAD is zero, which may produce very large z-scores.

**Current Code**:
```python
if mad == 0:
    mad = np.finfo(float).eps
```

**Problem**: When all values are identical, dividing by epsilon produces extremely large z-scores, which may not be meaningful.

**Impact**: Low - this is a known limitation of z-scores with zero variance.

**Recommendation**: Consider returning zero z-scores or a special value when MAD is zero, or document this behavior.

---

### 11. **Stability Detector - Candidate Selection Logic**

**Location**: `hubscan/core/detectors/stability.py:91-98`

**Issue**: If `candidate_indices` is provided but empty, the detector will process nothing.

**Current Code**:
```python
if candidate_indices is None:
    candidate_indices = np.arange(N)
    
if len(candidate_indices) > self.candidates_top_x:
    candidate_indices = candidate_indices[:self.candidates_top_x]
```

**Problem**: If `candidate_indices` is an empty array, the detector returns all zeros without warning.

**Impact**: Low - this may be intentional behavior, but should be documented or warned.

**Recommendation**: Add warning when `len(candidate_indices) == 0`.

---

## Summary of Recommendations

### High Priority (Fix Immediately)
1. Fix stability score normalization (Issue #1)
2. Add empty query checks (Issues #2, #3, #6)
3. Fix hubness validation inconsistency (Issue #4)
4. Add empty document set check (Issue #5)

### Medium Priority (Fix Soon)
5. Make stability detector reproducible (Issue #7)
6. Add bounds checking in dedup detector (Issue #8)

### Low Priority (Consider for Future)
7. Improve entropy normalization clarity (Issue #9)
8. Document zero-variance z-score behavior (Issue #10)
9. Add warnings for edge cases (Issue #11)

---

## Testing Recommendations

1. **Add edge case tests**:
   - Empty queries array
   - Empty document set
   - Single document/query
   - Zero variance in hub_rate

2. **Add mathematical correctness tests**:
   - Verify stability scores are in [0, 1]
   - Verify hub_rate normalization
   - Verify entropy normalization

3. **Add validation tests**:
   - Test exact validation with same query subset
   - Test reproducibility with fixed seeds

---

## Conclusion

The codebase is generally well-structured, but contains several mathematical and edge case issues that should be addressed. The most critical issue is the incorrect stability score calculation, which could lead to misleading results. The edge case issues could cause runtime crashes in production environments.

