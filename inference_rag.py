import faiss
import numpy as np

from utills import load_test_bench


def retrieve_most_similar(predicted_label_vector, vector_index, reports, k=5):
    query_vector = np.array(predicted_label_vector).astype('float32')
    query_vector = query_vector / np.linalg.norm(query_vector)

    # Search
    _, indices = vector_index.search(query_vector.reshape(1, -1), k=k)  # get top 5 similar

    # Retrieve reports
    retrieved_reports = [reports[idx] for idx in indices[0]]
    return retrieved_reports


def main():
    radio_bench_val = load_test_bench()


if __name__ == '__main__':
    main()
