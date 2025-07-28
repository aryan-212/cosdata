import os
import time
from cosdata import Client
from cosdata.embedding import embed_texts
from tqdm.auto import tqdm
import beir.util
from beir.datasets.data_loader import GenericDataLoader
import pandas as pd
import numpy as np
from beir.retrieval.evaluation import EvaluateRetrieval
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import csv
from datetime import datetime

# Configuration
CPU_CORES = multiprocessing.cpu_count()
# QPS_WORKERS = min(CPU_CORES * 2, 32)  # Commented out
# EMBEDDING_WORKERS = min(CPU_CORES, 8)  # Commented out
# BATCH_WORKERS = 32  # Commented out
DIMENSION = 768
BATCH_SIZE = 64
TOP_K = 30

# Dataset configurations
DATASET_CONFIGS = {
    "arguana": {"dataset_id": "arguana", "collection_name": "arguana-demo", "embedding_cache": "arguana_embeddings.npz", "description": "Argument retrieval dataset"},
    "fiqa": {"dataset_id": "fiqa", "collection_name": "fiqa-demo", "embedding_cache": "fiqa_embeddings.npz", "description": "Financial question answering dataset"},
    "scidocs": {"dataset_id": "scidocs", "collection_name": "scidocs-demo", "embedding_cache": "scidocs_embeddings.npz", "description": "Scientific document retrieval"},
    "scifact": {"dataset_id": "scifact", "collection_name": "scifact-demo", "embedding_cache": "scifact_embeddings.npz", "description": "Scientific fact checking dataset"},
    "trec-covid": {"dataset_id": "trec-covid", "collection_name": "trec-covid-demo", "embedding_cache": "trec-covid_embeddings.npz", "description": "COVID-19 document retrieval"},
    "webis-touche2020": {"dataset_id": "webis-touche2020", "collection_name": "webis-touche2020-demo", "embedding_cache": "webis-touche2020_embeddings.npz", "description": "Argument retrieval"}
}


def select_dataset():
    """Display available datasets and get user selection"""
    print("Available datasets:")
    print("-" * 50)
    dataset_names = list(DATASET_CONFIGS.keys())
    for i, name in enumerate(dataset_names, 1):
        config = DATASET_CONFIGS[name]
        print(f"{i}. {name} ({config['description']})")
    print("-" * 50)
    
    while True:
        try:
            choice = input("Please select a dataset (enter number): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(dataset_names):
                selected_dataset = dataset_names[choice_num - 1]
                print(f"\nSelected dataset: {selected_dataset}")
                return selected_dataset
            else:
                print(f"Please enter a number between 1 and {len(dataset_names)}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)


def select_target_pid():
    """Interactive PID selection for cosdata server monitoring"""
    print("\n" + "="*50)
    print("COSDATA SERVER MONITORING")
    print("="*50)
    print("Enter the PID of the Cosdata server for performance metrics")
    print("(Leave empty to skip monitoring)")
    
    while True:
        try:
            pid_input = input("Enter PID: ").strip()
            if not pid_input:
                print("Skipping server monitoring")
                return "skip"
            
            target_pid = int(pid_input)
            try:
                process = psutil.Process(target_pid)
                print(f"Monitoring PID {target_pid} - {process.name()}")
                return target_pid
            except psutil.NoSuchProcess:
                print(f"Error: PID {target_pid} not found")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return "skip"
        except ValueError:
            print("Error: Please enter a valid number")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                return "skip"
        except KeyboardInterrupt:
            print("\nSkipping server monitoring")
            return "skip"


def ensure_beir_dataset_available(dataset_id, save_dir):
    """Ensure a BEIR dataset is downloaded and ready to use"""
    datasets_dir = "datasets"
    if not os.path.exists(datasets_dir):
        os.makedirs(datasets_dir)
    
    if os.path.exists(save_dir):
        corpus_file = os.path.join(save_dir, "corpus.jsonl")
        queries_file = os.path.join(save_dir, "queries.jsonl")
        qrels_folder = os.path.join(save_dir, "qrels")
        
        if (os.path.exists(corpus_file) and os.path.exists(queries_file) and os.path.exists(qrels_folder)):
            print(f"Dataset {dataset_id} is already available at {save_dir}")
            return save_dir
    
    print(f"Dataset {dataset_id} not found. Downloading...")
    base_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"
    data_path = beir.util.download_and_unzip(base_url.format(dataset_id), save_dir)
    print(f"Dataset {dataset_id} downloaded successfully to {data_path}")
    return data_path


def get_memory_usage(pid):
    """Get current memory usage in GB for a specific PID"""
    try:
        process = psutil.Process(pid)
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024 * 1024)
    except psutil.NoSuchProcess:
        return 0.0


def get_cpu_usage(pid):
    """Get current CPU usage percentage for a specific PID"""
    try:
        process = psutil.Process(pid)
        return process.cpu_percent(interval=1)
    except psutil.NoSuchProcess:
        return 0.0


def monitor_operation_resources(operation_func, target_pid, operation_name):
    """Monitor resource usage during an operation and return max/avg values"""
    memory_readings = []
    cpu_readings = []
    
    def monitored_operation():
        memory_readings.append(get_memory_usage(target_pid))
        cpu_readings.append(get_cpu_usage(target_pid))
        
        result = operation_func()
        
        memory_readings.append(get_memory_usage(target_pid))
        cpu_readings.append(get_cpu_usage(target_pid))
        
        return result
    
    result = monitored_operation()
    
    if memory_readings and cpu_readings:
        max_memory = max(memory_readings)
        avg_memory = sum(memory_readings) / len(memory_readings)
        max_cpu = max(cpu_readings)
        avg_cpu = sum(cpu_readings) / len(cpu_readings)
        
        print(f"{operation_name} Resource Usage:")
        print(f"  Max Memory: {max_memory:.2f} GB")
        print(f"  Avg Memory: {avg_memory:.2f} GB")
        print(f"  Max CPU: {max_cpu:.1f}%")
        print(f"  Avg CPU: {avg_cpu:.1f}%")
        
        return max_memory, avg_memory, max_cpu, avg_cpu
    
    return 0.0, 0.0, 0.0, 0.0


def monitored_upsert_operation(collection, all_ids, all_embeddings, corpus_list):
    """Upsert vectors with monitoring"""
    print(f"Upserting {len(all_ids)} vectors in a single transaction...")
    with collection.transaction() as txn:
        for i in tqdm(range(0, len(all_ids), BATCH_SIZE), desc="Upsert"):
            batch_ids = all_ids[i : i + BATCH_SIZE]
            batch_embeddings = all_embeddings[i : i + BATCH_SIZE]
            batch_texts = [corpus_list[j]["text"] for j in range(len(batch_ids))]
            records = [
                {
                    "id": batch_ids[j],
                    "dense_values": batch_embeddings[j].tolist(),
                    "text": batch_texts[j],
                }
                for j in range(len(batch_ids))
            ]
            txn.batch_upsert_vectors(records)
        transaction_id = txn.transaction_id
    
    print("All records upserted. Waiting for transaction to complete...")
    final_status, success = txn.poll_completion(
        target_status="complete", max_attempts=30, sleep_interval=10
    )
    if not success:
        print(f"Warning: Transaction may not have completed. Final status: {final_status}")
    else:
        print("Transaction completed successfully.")
    
    return success


def monitored_search_operation(collection, queries_with_gt):
    """Run hybrid search with monitoring"""
    return batch_hybrid_search_evaluation(collection, queries_with_gt)


def wait_for_index_ready(collection, expected_count):
    """Wait for indexing to complete"""
    print("Waiting for indexing to complete...")
    for _ in range(30):  # Wait up to 5 minutes
        status = collection.indexing_status()
        if status.get("total_records_indexed_completed") == expected_count:
            print(f"Indexing complete: {expected_count} records indexed.")
            return True
        time.sleep(10)
    print("Warning: Indexing may not have completed")
    return False


def batch_hybrid_search_evaluation(collection, queries_with_gt):
    """Batch hybrid search for NDCG/Recall evaluation"""
    print(f"Running batch hybrid search for {len(queries_with_gt)} queries...")
    results = {}
    
    # Prepare batch queries
    batch_queries = []
    query_id_mapping = []
    
    for query_doc in queries_with_gt:
        try:
            query_text = query_doc["text"]
            query_vector = embed_texts([query_text], model_name="thenlper/gte-base")[0]
            query_terms = [[i, 1.0] for i, _ in enumerate(query_text.split())]
            
            batch_query = {
                "query_vector": query_vector,
                "query_terms": query_terms,
                "sparse_early_terminate_threshold": 0.1
            }
            
            batch_queries.append(batch_query)
            query_id_mapping.append(query_doc["_id"])
            
        except Exception as e:
            print(f"Failed to prepare query {query_doc['_id']}: {e}")
            results[query_doc["_id"]] = {}
    
    if not batch_queries:
        print("No valid queries to process")
        return results
    
    # Process in batches to avoid memory issues
    batch_size = 100  # Process 100 queries at a time
    all_results = {}
    
    for i in tqdm(range(0, len(batch_queries), batch_size), desc="Batch hybrid search"):
        batch = batch_queries[i:i + batch_size]
        batch_ids = query_id_mapping[i:i + batch_size]
        
        try:
            # Use the batch_hybrid function from the SDK
            batch_response = collection.search.batch_hybrid(
                queries=batch,
                top_k=TOP_K,
                fusion_constant_k=60.0,
                return_raw_text=True
            )
            
            # Process results
            if "responses" in batch_response:
                for j, response in enumerate(batch_response["responses"]):
                    query_id = batch_ids[j]
                    if "results" in response:
                        all_results[query_id] = {r["id"]: r["score"] for r in response["results"]}
                    else:
                        all_results[query_id] = {}
            else:
                print(f"Unexpected batch response format: {batch_response}")
                for query_id in batch_ids:
                    all_results[query_id] = {}
                    
        except Exception as e:
            print(f"Batch search failed for batch {i//batch_size}: {e}")
            for query_id in batch_ids:
                all_results[query_id] = {}
    
    # Merge results
    results.update(all_results)
    
    return results





# def calculate_latency_statistics(latencies):
#     """Calculate latency statistics"""
#     if not latencies:
#         return 0.0, 0.0, 0.0, 0.0, 0.0, 0
# 
#     latencies.sort()
#     p50_latency = latencies[int(len(latencies) * 0.5)]
#     p95_latency = latencies[int(len(latencies) * 0.95)]
#     min_latency = min(latencies)
#     max_latency = max(latencies)
#     avg_latency = sum(latencies) / len(latencies)
#     
#     print(f"Latency Statistics:")
#     print(f"p50 latency: {p50_latency:.2f} ms")
#     print(f"p95 latency: {p95_latency:.2f} ms")
#     print(f"Min latency: {min_latency:.2f} ms")
#     print(f"Max latency: {max_latency:.2f} ms")
#     print(f"Avg latency: {avg_latency:.2f} ms")
#     print(f"Total measurements: {len(latencies)}")
#     
#     return p50_latency, p95_latency, min_latency, max_latency, avg_latency, len(latencies)


# def run_qps_test(collection, queries, top_k=10, batch_size=100):
#     """Run QPS test"""
#     print(f"Running QPS test with {len(queries)} queries...")
#     
#     start_time = time.perf_counter()
#     results = []
#     
#     with ThreadPoolExecutor(max_workers=QPS_WORKERS) as executor:
#         futures = []
#         for i in range(0, len(queries), batch_size):
#             batch = queries[i:i + batch_size]
#             futures.append(executor.submit(batch_hybrid_search, collection, batch, top_k))
#         
#         for future in as_completed(futures):
#             try:
#                 future.result()
#                 results.append(True)
#             except Exception as e:
#                 print(f"Error in QPS test: {e}")
#                 results.append(False)
#     
#     end_time = time.perf_counter()
#     duration = end_time - start_time
#     total_queries = len(results) * batch_size
#     successful_queries = sum(results) * batch_size
#     qps = successful_queries / duration
#     
#     print(f"QPS: {qps:.2f}")
#     return qps, duration, successful_queries, total_queries - successful_queries, total_queries


# def batch_hybrid_search(collection, queries, top_k):
#     """Batch hybrid search"""
#     with ThreadPoolExecutor(max_workers=BATCH_WORKERS) as executor:
#         futures = []
#         for query in queries:
#             future = executor.submit(single_hybrid_search, collection, query, top_k)
#             futures.append(future)
#         
#         for future in as_completed(futures):
#             try:
#                 future.result()
#             except Exception as e:
#                 print(f"Batch search failed: {e}")
# 
# 
# def single_hybrid_search(collection, query, top_k):
#     """Single hybrid search"""
#     try:
#         query_text = query["text"]
#         query_vector = embed_texts([query_text], model_name="thenlper/gte-base")[0]
#         query_terms = [[i, 1.0] for i, _ in enumerate(query_text.split())]
#         hybrid_payload = {"query_vector": query_vector, "query_terms": query_terms}
#         collection.search.hybrid_search(hybrid_payload)
#     except Exception as e:
#         print(f"Single search failed: {e}")


def display_system_info(target_pid):
    """Display system information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"CPU Cores: {CPU_CORES}")
    # print(f"QPS Workers: {QPS_WORKERS}")  # Commented out
    # print(f"Batch Workers: {BATCH_WORKERS}")  # Commented out
    # print(f"Embedding Workers: {EMBEDDING_WORKERS}")  # Commented out
    
    if target_pid != "skip":
        try:
            process = psutil.Process(target_pid)
            print(f"Monitoring PID: {target_pid} - {process.name()}")
        except psutil.NoSuchProcess:
            print(f"PID {target_pid} not found")
    
    print("="*50)


def main():
    # Get user selection for dataset
    selected_dataset = select_dataset()
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIGS[selected_dataset]
    dataset_id = dataset_config["dataset_id"]
    collection_name = dataset_config["collection_name"]
    embedding_cache_path = dataset_config["embedding_cache"]
    
    print(f"Dataset ID: {dataset_id}")
    print(f"Collection name: {collection_name}")
    
    # Get target PID for cosdata server monitoring
    target_pid = select_target_pid()
    
    # Display system information
    display_system_info(target_pid)
    
    # 1. Download and load dataset using BEIR
    print(f"Downloading and loading {selected_dataset} dataset using BEIR...")
    save_dir = f"datasets/beir_dataset_{dataset_id}"
    data_path = ensure_beir_dataset_available(dataset_id, save_dir)
    
    # Load data using BEIR's GenericDataLoader
    loader = GenericDataLoader(data_folder=data_path)
    loader.check(fIn=loader.corpus_file, ext="jsonl")
    loader.check(fIn=loader.query_file, ext="jsonl")
    loader._load_corpus()
    loader._load_queries()
    
    # Determine split (msmarco uses 'dev', others use 'test')
    if dataset_id == "msmarco":
        split = "dev"
    else:
        split = "test"
    
    # Load qrels
    loader.qrels_file = os.path.join(loader.qrels_folder, split + ".tsv")
    loader._load_qrels()
    
    # Get corpus, queries, and qrels
    corpus = loader.corpus
    queries = loader.queries
    qrels = loader.qrels
    
    # Convert to format expected by the rest of the script
    corpus_list = [{"_id": k, "text": v["title"] + " " + v["text"]} for k, v in corpus.items()]
    
    # Handle different query formats in BEIR datasets
    queries_list = []
    for k, v in queries.items():
        if isinstance(v, dict):
            # Some datasets have queries as dictionaries with 'text' field
            queries_list.append({"_id": k, "text": v["text"]})
        else:
            # Some datasets have queries as strings directly
            queries_list.append({"_id": k, "text": v})
    
    print(f"Loaded {len(corpus_list)} corpus docs, {len(queries_list)} queries, {len(qrels)} qrels.")

    # 3. Prepare Cosdata client and always create collection and indexes
    client = Client()
    print(f"Creating collection '{collection_name}'...")
    collection = client.create_collection(collection_name, dimension=DIMENSION)
    collection.create_index(distance_metric="cosine")
    collection.create_tf_idf_index(name="tf_idf_index")
    collection.create_sparse_index(name="sparse_index")

    # 4. Load or compute embeddings
    if os.path.exists(embedding_cache_path):
        print(f"Loading cached embeddings from {embedding_cache_path}...")
        cache = np.load(embedding_cache_path, allow_pickle=True)
        all_ids = cache["ids"].tolist()
        all_embeddings = cache["embeddings"]
    else:
        print("Computing embeddings for corpus and caching them...")
        # print(f"Using {EMBEDDING_WORKERS} workers for parallel embedding generation")  # Commented out
        
        all_ids = []
        all_embeddings = []
        
        # Process embeddings in parallel batches
        with ThreadPoolExecutor(max_workers=8) as executor:  # Fixed number instead of EMBEDDING_WORKERS
            futures = []
            
            for i in range(0, len(corpus_list), BATCH_SIZE):
                batch = corpus_list[i : i + BATCH_SIZE]
                texts = [doc["text"] for doc in batch]
                ids = [doc["_id"] for doc in batch]
                
                # Submit batch for parallel processing
                future = executor.submit(embed_texts, texts, model_name="thenlper/gte-base")
                futures.append((future, ids))
            
            # Collect results with progress bar
            for future, ids in tqdm(futures, desc="Generating embeddings"):
                try:
                    embeddings = future.result()
                    all_ids.extend([str(x) for x in ids])
                    all_embeddings.extend(embeddings)
                except Exception as e:
                    print(f"Error generating embeddings for batch: {e}")
        
        all_embeddings = np.array(all_embeddings)
        np.savez_compressed(
            embedding_cache_path, ids=np.array(all_ids), embeddings=all_embeddings
        )
        print(f"Saved embeddings to {embedding_cache_path}.")

    # 5. Upsert all vectors in a single transaction (with monitoring)
    if target_pid != "skip":
        print("\n" + "="*50)
        print("MONITORING INSERTION PHASE")
        print("="*50)
        insertion_max_memory, insertion_avg_memory, insertion_max_cpu, insertion_avg_cpu = monitor_operation_resources(
            lambda: monitored_upsert_operation(collection, all_ids, all_embeddings, corpus_list),
            target_pid, "Insertion"
        )
    else:
        insertion_max_memory = insertion_avg_memory = insertion_max_cpu = insertion_avg_cpu = 0.0
        monitored_upsert_operation(collection, all_ids, all_embeddings, corpus_list)

    # 6. Wait for indexing to complete
    wait_for_index_ready(collection, expected_count=len(all_ids))
    print("Indexing complete.")

    # --- NDCG/Recall evaluation for hybrid search (with monitoring) ---
    # BEIR qrels are already in the correct format for EvaluateRetrieval
    print(f"Using {len(qrels)} qrels for evaluation.")

    # Prepare queries with ground truth
    valid_query_ids = set(qrels.keys())
    queries_with_gt = [q for q in queries_list if q["_id"] in valid_query_ids]

    # Run hybrid search for all queries and collect results (batch processing with monitoring)
    if target_pid != "skip":
        print("\n" + "="*50)
        print("MONITORING SEARCH PHASE")
        print("="*50)
        search_max_memory, search_avg_memory, search_max_cpu, search_avg_cpu = monitor_operation_resources(
            lambda: monitored_search_operation(collection, queries_with_gt),
            target_pid, "Search"
        )
        results = monitored_search_operation(collection, queries_with_gt)
    else:
        search_max_memory = search_avg_memory = search_max_cpu = search_avg_cpu = 0.0
        results = monitored_search_operation(collection, queries_with_gt)

    # Compute NDCG and Recall
    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(qrels, results, [1, 10])
    ndcg_score = ndcg["NDCG@10"]
    recall_score = recall["Recall@10"]
    print("Hybrid NDCG@10:", ndcg_score)
    print("Hybrid Recall@10:", recall_score)

    # # Calculate latency statistics from NDCG/Recall evaluation  # Commented out
    # p50_latency, p95_latency, min_latency, max_latency, avg_latency, total_measurements = calculate_latency_statistics(latencies)

    # # --- Performance Testing ---  # Commented out
    # # Get additional queries for performance testing (up to 10,000)
    # all_queries = [q for q in queries_list]
    # performance_queries = all_queries[:min(10000, len(all_queries))]
    # 
    # # Run performance tests
    # print("\n" + "="*50)
    # print("PERFORMANCE TESTING")
    # print("="*50)
    # 
    # # Run QPS test
    # qps, qps_duration, successful_queries, failed_queries, total_queries = run_qps_test(
    #     collection, performance_queries, top_k=10, batch_size=100
    # )
    
    # Display comprehensive results
    print("\n" + "="*50)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*50)
    print(f"Dataset: {selected_dataset}")
    print(f"Corpus Size: {len(corpus_list):,}")
    print(f"Queries Size: {len(queries_list):,}")
    print(f"Qrels Size: {len(qrels):,}")
    print()
    print("EVALUATION METRICS:")
    print(f"  NDCG@10: {ndcg_score:.4f}")
    print(f"  Recall@10: {recall_score:.4f}")
    print(f"  Search Method: Batch Hybrid Search")
    # print()  # Commented out
    # print("PERFORMANCE METRICS:")  # Commented out
    # print(f"  QPS: {qps:.2f}")  # Commented out
    # print(f"  p50 Latency: {p50_latency:.2f} ms")  # Commented out
    # print(f"  p95 Latency: {p95_latency:.2f} ms")  # Commented out
    # print(f"  Min Latency: {min_latency:.2f} ms")  # Commented out
    # print(f"  Max Latency: {max_latency:.2f} ms")  # Commented out
    # print(f"  Avg Latency: {avg_latency:.2f} ms")  # Commented out
    # print(f"  Latency Measurements: {total_measurements}")  # Commented out
    # print()  # Commented out
    if target_pid != "skip":
        print("RESOURCE USAGE (INSERTION):")
        print(f"  Max Memory: {insertion_max_memory:.2f} GB")
        print(f"  Avg Memory: {insertion_avg_memory:.2f} GB")
        print(f"  Max CPU: {insertion_max_cpu:.1f}%")
        print(f"  Avg CPU: {insertion_avg_cpu:.1f}%")
        print()
        print("RESOURCE USAGE (SEARCH):")
        print(f"  Max Memory: {search_max_memory:.2f} GB")
        print(f"  Avg Memory: {search_avg_memory:.2f} GB")
        print(f"  Max CPU: {search_max_cpu:.1f}%")
        print(f"  Avg CPU: {search_avg_cpu:.1f}%")
        print()
    # print("QPS TEST DETAILS:")  # Commented out
    # print(f"  Total Queries: {total_queries:,}")  # Commented out
    # print(f"  Successful Queries: {successful_queries:,}")  # Commented out
    # print(f"  Failed Queries: {failed_queries:,}")  # Commented out
    # print(f"  Test Duration: {qps_duration:.2f} seconds")  # Commented out
    # print(f"  Success Rate: {(successful_queries / total_queries * 100):.2f}%")  # Commented out
    print("="*50)
    
    # Return results in structured format
    results = {
        "dataset": selected_dataset,
        "corpus_size": len(corpus_list),
        "queries_size": len(queries_list),
        "qrels_size": len(qrels),
        "ndcg@10": ndcg_score,
        "recall@10": recall_score,
        "search_method": "batch_hybrid",
        # "qps": qps,  # Commented out
        # "p50_latency": p50_latency,  # Commented out
        # "p95_latency": p95_latency,  # Commented out
        # "min_latency": min_latency,  # Commented out
        # "max_latency": max_latency,  # Commented out
        # "avg_latency": avg_latency,  # Commented out
        # "latency_measurements": total_measurements,  # Commented out
        "insertion_max_memory": insertion_max_memory,
        "insertion_avg_memory": insertion_avg_memory,
        "insertion_max_cpu": insertion_max_cpu,
        "insertion_avg_cpu": insertion_avg_cpu,
        "search_max_memory": search_max_memory,
        "search_avg_memory": search_avg_memory,
        "search_max_cpu": search_max_cpu,
        "search_avg_cpu": search_avg_cpu,
        # "total_queries": total_queries,  # Commented out
        # "successful_queries": successful_queries,  # Commented out
        # "failed_queries": failed_queries,  # Commented out
        # "qps_duration": qps_duration,  # Commented out
        # "success_rate": (successful_queries / total_queries * 100)  # Commented out
    }
    
    return results


def save_results_to_csv(results, filename=None):
    """Save results to CSV file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"reciprocal_search_results_{timestamp}.csv"
    
    fieldnames = [
        "dataset", "corpus_size", "queries_size", "qrels_size",
        "ndcg@10", "recall@10", "search_method",
        # "qps", "p50_latency", "p95_latency",  # Commented out
        # "min_latency", "max_latency", "avg_latency", "latency_measurements",  # Commented out
        "insertion_max_memory", "insertion_avg_memory", "insertion_max_cpu", "insertion_avg_cpu",
        "search_max_memory", "search_avg_memory", "search_max_cpu", "search_avg_cpu",
        # "total_queries", "successful_queries", "failed_queries", "qps_duration", "success_rate"  # Commented out
    ]
    
    labels = [
        "Dataset", "Corpus Size", "Queries Size", "Qrels Size",
        "NDCG@10", "Recall@10", "Search Method",
        # "QPS", "p50 Latency (ms)", "p95 Latency (ms)",  # Commented out
        # "Min Latency (ms)", "Max Latency (ms)", "Avg Latency (ms)", "Latency Measurements",  # Commented out
        "Insertion Max Memory (GB)", "Insertion Avg Memory (GB)", "Insertion Max CPU (%)", "Insertion Avg CPU (%)",
        "Search Max Memory (GB)", "Search Avg Memory (GB)", "Search Max CPU (%)", "Search Avg CPU (%)",
        # "Total Queries", "Successful Queries", "Failed Queries", "QPS Duration (s)", "Success Rate (%)"  # Commented out
    ]
    
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header with labels
        writer.writerow({fieldname: label for fieldname, label in zip(fieldnames, labels)})
        
        # Write results
        writer.writerow(results)
    
    print(f"\nResults saved to: {filename}")
    return filename


if __name__ == "__main__":
    try:
        # Run the test and get results
        results = main()
        
        # Save results to CSV
        if results:
            save_results_to_csv(results)
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        raise
