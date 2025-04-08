
from neo4j import Driver
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

    def run(self, question: str, driver: Driver, openai_api_key):
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
        answer = self.llm2.generate_answer(question=question, nodes_data=retrieved_nodes_data)
        print("    ", f"Answer: {answer}")
        return answer