import argparse
import os

import torch
from datasets import load_from_disk
from dotenv import load_dotenv
from neo4j import GraphDatabase
from transformers import AutoTokenizer

from retrieval.retriever import Retriever

load_dotenv('.env', override=True)
NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

EF = 10_000
CYPHER_RATE = 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--max_nodes', type=int, default=None)
    parser.add_argument('--max_tokens', type=int, default=None)
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()

    match args.dataset:
        case 'prime':
            node_properties = ['name']  # ['name', 'details']
            sorting_index = 'textEmbedding'
            vector_index = 'textEmbedding'
        case 'mag':
            node_properties = ['name']  # ['name','abstract']
            sorting_index = 'nameEmbedding'  # Is actually the abstract embedding for papers
            vector_index = 'abstractEmbedding'
        case _:
            raise Exception('Unrecognized dataset name')

    if args.max_nodes is None:
        if args.max_tokens is None:
            raise Exception('Must specify --max_nodes or --max_tokens')
        else:
            count_tokens = True
            tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct')
    else:
        if args.max_tokens is not None:
            raise Exception('Cannot specify both --max_nodes and --max_tokens')
        else:
            count_tokens = False
            tokenizer = None

    retriever = Retriever(node_properties=node_properties, sorting_index=sorting_index, vector_index=vector_index,
                          pattern_rate=CYPHER_RATE, ef=EF,
                          count_tokens=count_tokens, max_nodes=args.max_nodes, max_tokens=args.max_tokens, formatter=None,
                          tokenizer=tokenizer)

    qa_with_generated_cypher_queries = load_from_disk(f"{args.dataset}-data/qa_with_generated_cypher_queries_gemma2_1/{args.split}")

    qa_with_generated_cypher_queries = qa_with_generated_cypher_queries.map(
        lambda x: x | {'top_cypher_queries': [q.replace('RETURN x', 'RETURN DISTINCT x') for q in x['top_cypher_queries']]}) #temp solution, to be removed

    q_embs = torch.load(f"{args.dataset}-data/text-embeddings-ada-002/query/query_emb_dict.pt", weights_only=False)

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        qa_with_retrieved_data = qa_with_generated_cypher_queries \
            .filter(lambda _, i: i % 1 == 0, with_indices=True) \
            .map(lambda x: x | {'q_emb': q_embs[x['id']].tolist()[0]}) \
            .map(lambda x: x | {
            'data': str(retriever.retrieve_data(driver=driver, cypher_queries=x['top_cypher_queries'], q_emb=x['q_emb']))},
                 num_proc=1,
                 cache_file_name=None)

    from compute_metrics import compute_metrics

    predss = [[data['nodeId'] for data in eval(datas_strs)] for datas_strs in qa_with_retrieved_data['data']]
    labelss = [x for x in qa_with_retrieved_data['answer_ids']]

    _ = compute_metrics(predss=predss, labelss=labelss,
                        metrics=['precision', 'recall', 'f1', 'hit@1', 'hit@5', 'recall@20', 'mrr', 'num_nodes'])


if __name__ == '__main__':
    main()