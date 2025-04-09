import unsloth
import argparse
from dotenv import load_dotenv
import os

from neo4j import GraphDatabase, Driver
import torch

from openai import OpenAI
from retrieval.ner import NER
from retrieval.path_retriever import PathRetriever
from llm1 import LLM1
from retrieval.retriever import Retriever
from llm2 import LLM2

class GraphRAFT:
    def __init__(self, dataset_name, llm1_model_dir, llm1_adapter_dir, llm1_beam_width, llm2_model_dir, llm2_max_sequence_length, openai_api_key):
        self.dataset_name = dataset_name
        match dataset_name:
            case 'prime':
                retrieve_properties = ['name', 'details']
                prompt_node_properties = ['pattern'] + retrieve_properties
                big_vector_index = 'textEmbedding'
            case 'mag':
                retrieve_properties = ['name', 'abstract']
                prompt_node_properties = ['pattern'] + retrieve_properties
                big_vector_index = 'abstractEmbedding'
            case _:
                raise Exception(f"Unknown dataset: {dataset_name}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ner = NER(dataset_name=dataset_name, openai_api_key=openai_api_key)
        self.path_retriever = PathRetriever(dataset_name)
        self.llm1 = LLM1(device=device, model_dir=llm1_model_dir, adapter_dir=llm1_adapter_dir, beam_width=llm1_beam_width)
        self.llm1.put_in_inference_mode()
        self.data_retriever = Retriever(node_properties=retrieve_properties, max_nodes=20, vector_index=big_vector_index)
        self.llm2 = LLM2(model_dir=llm2_model_dir, properties=prompt_node_properties, max_sequence_length=llm2_max_sequence_length)
        self.llm2.put_in_inference_mode()

    def run(self, question: str, driver: Driver, openai_api_key, add_more_answers=False):
        print(f"Question: {question}")
        question_embedding = OpenAI(api_key=openai_api_key).embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

        print("Predict source nodes...")
        predicted_source_nodes = self.ner.find_source_nodes(question=question, driver=driver)
        print("    ", predicted_source_nodes)

        print("Find all possible queries...")
        all_possible_queries = self.path_retriever.retrieve_paths(driver, predicted_source_nodes)['cypher_queries']
        print("    ", f"#: {len(all_possible_queries)}")

        print(f"Generate retrieval queries...")
        predicted_top_queries = self.llm1.predict_top_queries(question=question, possible_queries=all_possible_queries)
        #print("    ", f"Top queries: {predicted_top_queries}")
        for query in predicted_top_queries:
            print("    ", query)

        print(f"Retrieve data...")
        retrieved_nodes_data = self.data_retriever.retrieve_data(driver, cypher_queries=predicted_top_queries, q_emb=question_embedding)
        print("    ", f"#nodes: {len(retrieved_nodes_data)}")

        print(f"Generate answer...")
        answer = self.llm2.generate_answer(question=question, nodes_data=retrieved_nodes_data, add_more_answers=add_more_answers)
        print("    ", f"Answer: {answer}")
        return answer

def main():
    load_dotenv('db.env', override=True)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    NEO4J_URI = os.getenv('NEO4J_URI')
    NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
    NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--llm1_model_dir", type=str, default="neo4j/text2cypher-gemma-2-9b-it-finetuned-2024v1" )
    parser.add_argument("--llm1_adapter_dir", type=str, default=None)
    parser.add_argument("--llm1_beam_width", type=int, default=5)
    parser.add_argument("--llm2_model_dir", type=str, default=None)
    parser.add_argument("--llm2_max_sequence_length", type=int, default=11_000)
    parser.add_argument("--add_more_answers", action="store_true")
    args = parser.parse_args()

    dataset_name = args.dataset
    llm1_adapter_dir = f"{dataset_name}-models/llm1" if args.llm1_adapter_dir is None else args.llm1_adapter_dir
    llm2_model_dir = f"{dataset_name}-models/llm2" if args.llm2_model_dir is None else args.llm2_model_dir

    graph_raft = GraphRAFT(dataset_name=dataset_name, llm1_model_dir=args.llm1_model_dir,
                           llm1_adapter_dir=llm1_adapter_dir, llm1_beam_width=args.llm1_beam_width,
                           llm2_model_dir=llm2_model_dir, llm2_max_sequence_length=args.llm2_max_sequence_length,
                           openai_api_key=OPENAI_API_KEY)

    with GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)) as driver:
        while True:
            question = input("Question: ")
            graph_raft.run(question=question, driver=driver, openai_api_key=OPENAI_API_KEY, args.add_more_answers)

if __name__ == "__main__":
    main()