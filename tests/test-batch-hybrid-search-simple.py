import os
import time
import random
import numpy as np
from cosdata import Client
import requests


def create_test_collection():
    """Create a test collection with all required indexes"""
    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    client = Client(host=host)

    collection_name = "test_batch_hybrid_simple"

    # Try to get existing collection or create new one
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except:
        print("Creating new collection...")
        collection = client.create_collection(
            name=collection_name,
            description="Test collection for batch hybrid search",
            dimension=512,
            dense_vector={"enabled": True, "dimension": 512},
            sparse_vector={"enabled": True},
            tf_idf_options={"enabled": True},
        )
        print("Collection created successfully")

    # Create required indexes
    print("Creating indexes...")
    try:
        collection.create_index(
            distance_metric="cosine",
            num_layers=8,
            max_cache_size=1000,
            ef_construction=64,
            ef_search=64,
            neighbors_count=16,
            level_0_neighbors_count=32,
        )
        print("Dense index created")
    except Exception as e:
        print(f"Dense index may already exist: {e}")

    try:
        collection.create_sparse_index(
            name="sparse_index", quantization=64, sample_threshold=1000
        )
        print("Sparse index created")
    except Exception as e:
        print(f"Sparse index may already exist: {e}")

    try:
        collection.create_tf_idf_index(
            name="tf_idf_index", sample_threshold=1000, k1=1.5, b=0.75
        )
        print("TF-IDF index created")
    except Exception as e:
        print(f"TF-IDF index may already exist: {e}")

    return collection, client


def generate_test_data(num_vectors=50):
    """Generate test data with dense, sparse, and text content"""
    vectors = []

    for i in range(num_vectors):
        # Generate dense vector
        dense_vector = np.random.normal(0, 1, 512).tolist()

        # Generate sparse vector
        non_zero_dims = random.randint(20, 100)
        indices = sorted(random.sample(range(1000), non_zero_dims))
        values = np.random.uniform(0.0, 2.0, non_zero_dims).tolist()

        # Generate text content
        text_content = f"Document {i} with sample text content for TF-IDF indexing"

        vectors.append(
            {
                "id": f"vec_{i}",
                "dense_values": dense_vector,
                "sparse_indices": indices,
                "sparse_values": values,
                "metadata": {
                    "text": text_content,
                    "category": f"category_{i % 5}",
                },
            }
        )

    return vectors


def test_batch_hybrid_search(collection, client):
    """Test batch hybrid search functionality"""
    print("Testing batch hybrid search...")

    host = os.getenv("COSDATA_HOST", "http://127.0.0.1:8443")
    base_url = f"{host}/vectordb"
    collection_name = collection.name

    # Create test queries
    queries = [
        {
            "query_vector": np.random.normal(0, 1, 512).tolist(),
            "query_terms": [[i, random.uniform(0.1, 2.0)] for i in range(50, 100)],
            "sparse_early_terminate_threshold": 0.1,
        },
        {
            "query_vector": np.random.normal(0, 1, 512).tolist(),
            "query_text": "sample text content for TF-IDF search",
        },
        {
            "query_terms": [[i, random.uniform(0.1, 2.0)] for i in range(100, 150)],
            "query_text": "another text query for hybrid search",
            "sparse_early_terminate_threshold": 0.2,
        },
    ]

    payload = {
        "queries": queries,
        "top_k": 10,
        "fusion_constant_k": 60.0,
        "return_raw_text": True,
    }

    url = f"{base_url}/collections/{collection_name}/search/batch-hybrid"

    try:
        response = requests.post(
            url, json=payload, headers=client._get_headers(), verify=client.verify_ssl
        )

        if response.status_code == 200:
            results = response.json()
            print(
                f"✅ Batch hybrid search successful! Got {len(results['responses'])} responses"
            )
            for i, response_data in enumerate(results["responses"]):
                print(f"  Query {i + 1}: {len(response_data['results'])} results")
            return True
        else:
            print(f"❌ Batch hybrid search failed: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Batch hybrid search failed: {e}")
        return False


def main():
    """Main test function"""
    print("Starting simplified batch hybrid search test...")

    try:
        # Create test collection
        collection, client = create_test_collection()
        print(f"Created collection: {collection.name}")

        # Generate and insert test data
        print("Generating and inserting test data...")
        vectors = generate_test_data(50)

        with collection.transaction() as txn:
            txn.batch_upsert_vectors(vectors)

        # Wait for indexing
        print("Waiting for indexing to complete...")
        time.sleep(3)

        # Test batch hybrid search
        success = test_batch_hybrid_search(collection, client)

        if success:
            print("\n✅ Batch hybrid search test passed!")
            return 0
        else:
            print("\n❌ Batch hybrid search test failed!")
            return 1

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())

