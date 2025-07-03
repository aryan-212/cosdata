use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::OnceLock;

/// Returns the largest power of 4 that is less than or equal to `n`.
/// Iteratively multiplies by 4 until the result exceeds `n`.
pub fn largest_power_of_4_below(n: u32) -> (u8, u32) {
    assert_ne!(n, 0, "Cannot find largest power of 4 below 0");
    let msb_position = (31 - n.leading_zeros()) as u8;
    let power = msb_position / 2;
    let value = 1u32 << (power * 2);
    (power, value)
}

/// Calculates the path from `current_dim_index` to `target_dim_index`.
/// Decomposes the difference into powers of 4 and returns the indices.
pub fn calculate_path(target_dim_index: u32, current_dim_index: u32) -> Vec<u8> {
    let mut path = Vec::new();
    let mut remaining = target_dim_index - current_dim_index;

    while remaining > 0 {
        let (child_index, pow_4) = largest_power_of_4_below(remaining);
        path.push(child_index);
        remaining -= pow_4;
    }

    path
}

/// Global cache for path calculations to avoid recomputation
static PATH_CACHE: OnceLock<Mutex<HashMap<(u32, u32), Vec<u8>>>> = OnceLock::new();

/// DP-optimized version of calculate_path that caches results
/// This is particularly useful when the same path calculations are performed repeatedly
pub fn calculate_path_dp(target_dim_index: u32, current_dim_index: u32) -> Vec<u8> {
    // Get or initialize the global cache
    let cache = PATH_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    
    // Create cache key from the difference (since paths depend on the difference, not absolute values)
    let difference = target_dim_index - current_dim_index;
    let cache_key = (current_dim_index, target_dim_index);
    
    // Check if we have a cached result
    if let Ok(cache_guard) = cache.lock() {
        if let Some(cached_path) = cache_guard.get(&cache_key) {
            return cached_path.clone();
        }
    }
    
    // Calculate the path using the original algorithm
    let path = calculate_path(target_dim_index, current_dim_index);
    
    // Cache the result for future use
    if let Ok(mut cache_guard) = cache.lock() {
        cache_guard.insert(cache_key, path.clone());
        
        // Limit cache size to prevent memory bloat (keep only most recent 10000 entries)
        if cache_guard.len() > 10000 {
            // Simple LRU-like eviction: remove oldest entries
            let keys_to_remove: Vec<_> = cache_guard.keys().take(1000).cloned().collect();
            for key in keys_to_remove {
                cache_guard.remove(&key);
            }
        }
    }
    
    path
}

/// Clears the path cache to free memory
/// Useful when memory usage becomes a concern
pub fn clear_path_cache() {
    if let Some(cache) = PATH_CACHE.get() {
        if let Ok(mut cache_guard) = cache.lock() {
            cache_guard.clear();
        }
    }
}

/// Gets the current size of the path cache
pub fn get_path_cache_size() -> usize {
    if let Some(cache) = PATH_CACHE.get() {
        if let Ok(cache_guard) = cache.lock() {
            return cache_guard.len();
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_largest_power_of_4_below() {
        assert_eq!(largest_power_of_4_below(1), (0, 1));
        assert_eq!(largest_power_of_4_below(4), (1, 4));
        assert_eq!(largest_power_of_4_below(16), (2, 16));
        assert_eq!(largest_power_of_4_below(15), (1, 4));
        assert_eq!(largest_power_of_4_below(17), (2, 16));
    }

    #[test]
    fn test_calculate_path() {
        assert_eq!(calculate_path(1, 0), vec![0]);
        assert_eq!(calculate_path(4, 0), vec![1]);
        assert_eq!(calculate_path(16, 0), vec![2]);
        assert_eq!(calculate_path(17, 0), vec![2, 0]);
        assert_eq!(calculate_path(20, 0), vec![2, 1]);
    }

    #[test]
    fn test_calculate_path_dp() {
        // Clear cache before testing
        clear_path_cache();
        
        // Test basic functionality
        assert_eq!(calculate_path_dp(1, 0), vec![0]);
        assert_eq!(calculate_path_dp(4, 0), vec![1]);
        assert_eq!(calculate_path_dp(16, 0), vec![2]);
        assert_eq!(calculate_path_dp(17, 0), vec![2, 0]);
        assert_eq!(calculate_path_dp(20, 0), vec![2, 1]);
        
        // Test that cache is working
        let initial_cache_size = get_path_cache_size();
        assert!(initial_cache_size > 0);
        
        // Call the same function again - should use cache
        let result = calculate_path_dp(20, 0);
        assert_eq!(result, vec![2, 1]);
        
        // Cache size should remain the same (no new entries)
        assert_eq!(get_path_cache_size(), initial_cache_size);
    }

    #[test]
    fn test_cache_consistency() {
        clear_path_cache();
        
        // Test that DP version gives same results as original
        for target in 0..100 {
            for current in 0..target {
                let original_result = calculate_path(target, current);
                let dp_result = calculate_path_dp(target, current);
                assert_eq!(original_result, dp_result, 
                    "Mismatch for target={}, current={}", target, current);
            }
        }
    }
}
