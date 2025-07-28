#!/usr/bin/env python3
"""
Comprehensive test for batch hybrid search endpoint

This test covers:
- Collection and index setup
- Data population with dense, sparse, and text content
- All hybrid query types (DenseAndSparse, DenseAndTFIDF, SparseAndTFIDF)
- Response validation and error handling
- Performance benchmarking
"""

import os
import time
import random
import numpy as np
import requests
from cosdata import Client
from typing import List, Dict, Any
import json


class BatchHybridSearchTest:
    """Comprehensive test class for batch hybrid search functionality"""
    
    def __init__(self):
        self.host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
        self.username = os.getenv("COSDATA_USERNAME", "admin")
        self.password = os.getenv("COSDATA_PASSWORD", "admin")
        self.collection_name = "test_batch_hybrid_comprehensive"
        self.client = None
        self.collection = None
        
    def setup_client(self):
        """Initialize the cosdata client"""
        print("🔧 Setting up client...")
        self.client = Client(
            host=self.host,
            username=self.username,
            password=self.password,
            verify=False
        )
        print("✅ Client setup complete")
        
    def create_collection(self):
        """Create test collection with all required indexes"""
        print(f"🔧 Creating collection: {self.collection_name}")
        
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(name=self.collection_name)
            print(f"✅ Using existing collection: {self.collection_name}")
        except Exception:
            print("📝 Creating new collection...")
            self.collection = self.client.create_collection(
                name=self.collection_name,
                description="Comprehensive test collection for batch hybrid search",
                dimension=512,
                dense_vector={"enabled": True, "dimension": 512},
                sparse_vector={"enabled": True},
                tf_idf_options={"enabled": True},
            )
            print("✅ Collection created successfully")
            
    def create_indexes(self):
        """Create all required indexes for hybrid search"""
        print("🔧 Creating indexes...")
        
        # Create dense index
        try:
            self.collection.create_index(
                distance_metric="cosine",
                num_layers=8,
                max_cache_size=1000,
                ef_construction=64,
                ef_search=64,
                neighbors_count=16,
                level_0_neighbors_count=32,
            )
            print("✅ Dense index created")
        except Exception as e:
            print(f"ℹ️  Dense index may already exist: {e}")
            
        # Create sparse index
        try:
            self.collection.create_sparse_index(
                name="sparse_index",
                quantization=64,
                sample_threshold=1000
            )
            print("✅ Sparse index created")
        except Exception as e:
            print(f"ℹ️  Sparse index may already exist: {e}")
            
        # Create TF-IDF index
        try:
            self.collection.create_tf_idf_index(
                name="tf_idf_index",
                sample_threshold=1000,
                k1=1.5,
                b=0.75
            )
            print("✅ TF-IDF index created")
        except Exception as e:
            print(f"ℹ️  TF-IDF index may already exist: {e}")
            
    def generate_test_data(self, num_vectors: int = 100) -> List[Dict[str, Any]]:
        """Generate comprehensive test data with dense, sparse, and text content"""
        print(f"🔧 Generating {num_vectors} test vectors...")
        
        vectors = []
        categories = ["technology", "science", "art", "history", "sports"]
        
        for i in range(num_vectors):
            # Generate dense vector
            dense_vector = np.random.normal(0, 1, 512).tolist()
            
            # Generate sparse vector (simulate bag-of-words)
            non_zero_dims = random.randint(20, 100)
            indices = sorted(random.sample(range(1000), non_zero_dims))
            values = np.random.uniform(0.1, 2.0, non_zero_dims).tolist()
            
            # Format sparse vector as pairs [index, value]
            sparse_pairs = [[indices[i], values[i]] for i in range(len(indices))]
            
            # Generate realistic text content
            category = categories[i % len(categories)]
            text_content = f"Document {i} about {category}. "
            text_content += f"This document contains information about {category} topics. "
            text_content += f"The content includes various aspects of {category} and related subjects. "
            text_content += f"Sample text for TF-IDF indexing and search functionality."
            
            vectors.append({
                "id": f"vec_{i}",
                "dense_values": dense_vector,
                "sparse_values": sparse_pairs,
                "metadata": {
                    "text": text_content,
                    "category": category,
                    "document_id": f"doc_{i}",
                    "tags": [category, f"tag_{i % 10}"]
                }
            })
            
        print(f"✅ Generated {len(vectors)} test vectors")
        return vectors
        
    def populate_data(self, vectors: List[Dict[str, Any]]):
        """Insert test data into collection"""
        print("🔧 Inserting test data...")
        
        with self.collection.transaction() as txn:
            txn.batch_upsert_vectors(vectors)
            
        print("✅ Data insertion complete")
        
        # Wait for indexing
        print("⏳ Waiting for indexing to complete...")
        time.sleep(5)
        
    def create_test_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive test queries covering all hybrid search types"""
        print("🔧 Creating test queries...")
        
        queries = [
            # Query 1: DenseAndSparse
            {
                "DenseAndSparse": {
                    "query_vector": np.random.normal(0, 1, 512).tolist(),
                    "query_terms": [[i, random.uniform(0.1, 2.0)] for i in range(50, 100)],
                    "sparse_early_terminate_threshold": 0.1
                }
            },
            
            # Query 2: DenseAndTFIDF
            {
                "DenseAndTFIDF": {
                    "query_vector": np.random.normal(0, 1, 512).tolist(),
                    "query_text": "technology science research development"
                }
            },
            
            # Query 3: SparseAndTFIDF
            {
                "SparseAndTFIDF": {
                    "query_terms": [[i, random.uniform(0.1, 2.0)] for i in range(100, 150)],
                    "query_text": "art history culture museum",
                    "sparse_early_terminate_threshold": 0.2
                }
            },
            
            # Query 4: Another DenseAndSparse
            {
                "DenseAndSparse": {
                    "query_vector": np.random.normal(0, 1, 512).tolist(),
                    "query_terms": [[i, random.uniform(0.1, 2.0)] for i in range(200, 250)],
                    "sparse_early_terminate_threshold": 0.15
                }
            },
            
            # Query 5: Another DenseAndTFIDF
            {
                "DenseAndTFIDF": {
                    "query_vector": np.random.normal(0, 1, 512).tolist(),
                    "query_text": "sports athletics competition performance"
                }
            }
        ]
        
        print(f"✅ Created {len(queries)} test queries")
        return queries
        
    def test_batch_hybrid_search(self, queries: List[Dict[str, Any]]) -> bool:
        """Test the batch hybrid search endpoint"""
        print("🔧 Testing batch hybrid search...")
        
        payload = {
            "queries": queries,
            "top_k": 10,
            "fusion_constant_k": 60.0,
            "return_raw_text": True
        }
        
        url = f"{self.host}/vectordb/collections/{self.collection_name}/search/batch-hybrid"
        
        try:
            start_time = time.time()
            response = requests.post(
                url,
                json=payload,
                headers=self.client._get_headers(),
                verify=False
            )
            end_time = time.time()
            
            if response.status_code == 200:
                results = response.json()
                print(f"✅ Batch hybrid search successful!")
                print(f"⏱️  Response time: {end_time - start_time:.3f} seconds")
                print(f"📊 Got {len(results['responses'])} responses")
                
                # Validate response structure
                self._validate_response(results, len(queries))
                
                # Print detailed results
                self._print_results(results)
                
                return True
            else:
                print(f"❌ Batch hybrid search failed:")
                print(f"   Status code: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Batch hybrid search failed with exception: {e}")
            return False
            
    def _validate_response(self, results: Dict[str, Any], expected_queries: int):
        """Validate the response structure"""
        print("🔍 Validating response structure...")
        
        # Check if responses key exists
        if "responses" not in results:
            raise ValueError("Response missing 'responses' key")
            
        # Check number of responses matches queries
        if len(results["responses"]) != expected_queries:
            raise ValueError(f"Expected {expected_queries} responses, got {len(results['responses'])}")
            
        # Validate each response
        for i, response in enumerate(results["responses"]):
            if "results" not in response:
                raise ValueError(f"Response {i} missing 'results' key")
                
            if not isinstance(response["results"], list):
                raise ValueError(f"Response {i} 'results' is not a list")
                
        print("✅ Response structure validation passed")
        
    def _print_results(self, results: Dict[str, Any]):
        """Print detailed results for analysis"""
        print("\n📋 Detailed Results:")
        print("=" * 50)
        
        for i, response in enumerate(results["responses"]):
            print(f"\nQuery {i + 1}:")
            print(f"  Results: {len(response['results'])} items")
            
            if response["results"]:
                # Show top 3 results
                for j, result in enumerate(response["results"][:3]):
                    print(f"    {j + 1}. ID: {result['id']}, Score: {result['score']:.4f}")
                    if result.get("text"):
                        print(f"       Text: {result['text'][:100]}...")
                        
        if "warning" in results and results["warning"]:
            print(f"\n⚠️  Warning: {results['warning']}")
            
    def test_error_handling(self) -> bool:
        """Test error handling with invalid requests"""
        print("🔧 Testing error handling...")
        
        # Test with invalid collection
        url = f"{self.host}/vectordb/collections/nonexistent_collection/search/batch-hybrid"
        payload = {
            "queries": [{"DenseAndSparse": {"query_vector": [0.1, 0.2], "query_terms": []}}],
            "top_k": 10
        }
        
        try:
            response = requests.post(
                url,
                json=payload,
                headers=self.client._get_headers(),
                verify=False
            )
            
            if response.status_code == 404:
                print("✅ Error handling test passed (404 for nonexistent collection)")
                return True
            else:
                print(f"❌ Unexpected status code: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
            return False
            
    def run_performance_test(self, queries: List[Dict[str, Any]], iterations: int = 5) -> Dict[str, float]:
        """Run performance benchmark"""
        print(f"🔧 Running performance test ({iterations} iterations)...")
        
        times = []
        url = f"{self.host}/vectordb/collections/{self.collection_name}/search/batch-hybrid"
        payload = {
            "queries": queries,
            "top_k": 10,
            "fusion_constant_k": 60.0,
            "return_raw_text": True
        }
        
        for i in range(iterations):
            start_time = time.time()
            response = requests.post(
                url,
                json=payload,
                headers=self.client._get_headers(),
                verify=False
            )
            end_time = time.time()
            
            if response.status_code == 200:
                times.append(end_time - start_time)
                print(f"  Iteration {i + 1}: {times[-1]:.3f}s")
            else:
                print(f"  Iteration {i + 1}: Failed")
                
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"📊 Performance Results:")
            print(f"  Average: {avg_time:.3f}s")
            print(f"  Min: {min_time:.3f}s")
            print(f"  Max: {max_time:.3f}s")
            
            return {
                "average": avg_time,
                "min": min_time,
                "max": max_time,
                "iterations": len(times)
            }
        else:
            print("❌ Performance test failed - no successful iterations")
            return {}
            
    def run_comprehensive_test(self) -> bool:
        """Run the complete comprehensive test"""
        print("🚀 Starting comprehensive batch hybrid search test...")
        print("=" * 60)
        
        try:
            # Setup
            self.setup_client()
            self.create_collection()
            self.create_indexes()
            
            # Generate and populate data
            vectors = self.generate_test_data(100)
            self.populate_data(vectors)
            
            # Create test queries
            queries = self.create_test_queries()
            
            # Test basic functionality
            print("\n" + "=" * 60)
            print("🔧 Testing basic batch hybrid search functionality...")
            basic_success = self.test_batch_hybrid_search(queries)
            
            if not basic_success:
                print("❌ Basic functionality test failed")
                return False
                
            # Test error handling
            print("\n" + "=" * 60)
            print("🔧 Testing error handling...")
            error_success = self.test_error_handling()
            
            if not error_success:
                print("❌ Error handling test failed")
                return False
                
            # Run performance test
            print("\n" + "=" * 60)
            print("🔧 Running performance benchmark...")
            performance_results = self.run_performance_test(queries)
            
            if not performance_results:
                print("❌ Performance test failed")
                return False
                
            # Final summary
            print("\n" + "=" * 60)
            print("🎉 COMPREHENSIVE TEST SUMMARY")
            print("=" * 60)
            print("✅ Basic functionality: PASSED")
            print("✅ Error handling: PASSED")
            print("✅ Performance test: PASSED")
            print(f"📊 Performance: {performance_results['average']:.3f}s avg")
            print("🎯 All tests completed successfully!")
            
            return True
            
        except Exception as e:
            print(f"❌ Comprehensive test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Main test function"""
    test = BatchHybridSearchTest()
    success = test.run_comprehensive_test()
    
    if success:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main()) 