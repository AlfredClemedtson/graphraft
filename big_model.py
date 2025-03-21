
from neo4j import Driver

from ner import NER
from path_retriever import PathRetriever
from llm1 import LLM1
from retriever import Retriever
from llm2 import LLM2

class BigModel:
    def __init__(self, driver, dataset_name, llm1_model_dir, llm2_model_dir, openai_api_key):
        self.dataset_name = dataset_name
        match dataset_name:
            case 'prime':
                max_sequence_length = 15_000#PRIME_MAX_SEQUENCE_LENGTH  # calculate instead
                properties = ['pattern', 'description']
            case 'mag':
                max_sequence_length = 10_000#MAG_MAX_SEQUENCE_LENGTH
                properties = ['pattern', 'name', 'abstract']
            case _:
                raise Exception(f"Unknown dataset: {dataset_name}")

        self.ner = NER(driver, dataset_name, openai_api_key)
        self.path_retriever = PathRetriever(dataset_name)
        self.llm1 = LLM1(dataset_name, llm1_model_dir)
        self.data_retriever = Retriever(dataset_name)
        self.llm2 = LLM2(llm2_model_dir, properties, max_sequence_length)

    def run(self, question: str, driver: Driver):
        print(f"Question: {question}")
        question_embedding = None

        print("Predict source nodes...")
        predicted_source_nodes = self.ner.find_source_nodes(question=question, driver=driver)
        print(predicted_source_nodes)

        print("Find all possible queries...")
        all_possible_queries = self.path_retriever.retrieve_paths(driver, predicted_source_nodes)
        print(f"#: {len(all_possible_queries)}")

        print(f"Generate retrieval queries...")
        predicted_top_queries = self.llm1.generate_top_queries(question=question, all_possible_queries=all_possible_queries)
        print(f"Top queries: {predicted_top_queries}")

        print(f"Retrieve data...")
        retrieved_nodes_data = self.data_retriever.retrieve_data(driver, cypher_queries=predicted_top_queries, q_emb=question_embedding)
        print(f"#nodes: {len(retrieved_nodes_data)}")

        print(f"Generate answer...")
        answer = self.llm2.generate_answer(question=question, nodes_data=retrieved_nodes_data)
        print(f"Answer: {answer}")
        return answer