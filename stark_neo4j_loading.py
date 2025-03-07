from stark_qa.skb import SKB
from neo4j import GraphDatabase
from tqdm import tqdm
import torch
import numpy as np

def _chunks(xs, n: int = 10_000):
    n = max(1, n)
    return [xs[i:i + n] for i in range(0, len(xs), n)]

def _format_node_label(node_type: str) -> str:
    return "".join([s.capitalize() for s in node_type.replace('/','_or_').split('_')])

def _format_relationship_type(edge_type: str, dataset_name: str) -> str:
    if dataset_name == 'mag':
        return edge_type.split('___')[1].upper()
    else:
        return edge_type.replace(' ', '_').replace('-', '_').upper()

def insert_nodes(skb: SKB, driver: GraphDatabase.driver, key_property: str = None, ignore_properties: list[str] = None):
    # Require the nodeId to be unique for all nodes.
    driver.execute_query(
        f"""CREATE CONSTRAINT unique_nodeId IF NOT EXISTS FOR (n:_Entity_) REQUIRE n.nodeId IS UNIQUE""")

    for node_type in skb.node_type_dict.values():
        label = _format_node_label(node_type)  # field_of_study -> FieldOfStudy
        print(f'======  Loading {label} nodes  ======')

        nodes = [skb.node_info[node_id] |
                 {'key_id' : node_id if key_property is None else skb.node_info[node_id][key_property]}
                 for node_id in skb.get_node_ids_by_type(node_type)]
        #properties = set.union(*[set(node.keys()) for node in nodes]).difference(['key_id'] + ignore_properties)
        properties = set.union(*[set(node.keys()) for node in nodes]).difference(['key_id'])
        query = f"""UNWIND $recs AS rec
                    MERGE(n:{label}:_Entity_ {{nodeId: rec.key_id}})
                    SET {", ".join([f"n.{prop} = rec.{prop}" for prop in properties])}
                """
        with tqdm(total=len(nodes)) as pbar:  # Insert entities in db, one batch at a time
            for recs in _chunks(nodes, 5_000):
                driver.execute_query(query, parameters_={'recs': recs})
                pbar.update(len(recs))

def insert_relationships(skb: SKB, driver: GraphDatabase.driver, dataset_name: str):
    for edge_type in skb.edge_type_dict.values():
        edge_ids = skb.get_edge_ids_by_type(edge_type)

        rel_type = _format_relationship_type(edge_type, dataset_name)
        print(f'======  Loading {rel_type} edges  ======')

        query = f"""UNWIND $node_pairs AS nodePair
                    MATCH (src:_Entity_ {{nodeId: nodePair[0]}})
                    MATCH (tgt:_Entity_ {{nodeId: nodePair[1]}})
                    MERGE (src)-[:{rel_type}]->(tgt)"""

        node_pairs = skb.edge_index[:, edge_ids]
        num_edges = node_pairs.shape[1]
        with tqdm(total=num_edges) as pbar:
            for chunk in _chunks(np.arange(num_edges), 10_000):
                node_pairs_chunk = node_pairs[:, chunk].T.tolist()
                driver.execute_query(query, parameters_={'rel_type': rel_type, 'node_pairs': node_pairs_chunk})
                pbar.update(len(chunk))

def insert_node_embeddings(embeddings: dict[int, torch.Tensor], embedding_name: str, driver: GraphDatabase.driver):
    print(f"======  Loading embeddings ({embedding_name}) ======")
    paper_emb_records = [{"nodeId": idx, "textEmbedding": emb.squeeze().tolist()} for idx, emb in embeddings.items()]
    query = """UNWIND $recs AS rec
               MATCH (n:_Entity_ {nodeId: rec.nodeId})
               CALL db.create.setNodeVectorProperty(n, $embeddingName, rec.textEmbedding)"""
    with tqdm(total=len(paper_emb_records)) as pbar:
        for recs in _chunks(paper_emb_records, 1_000):
            driver.execute_query(query, parameters_={'recs': recs, 'embeddingName': embedding_name})
            pbar.update(len(recs))

        dim = 1536
        query = f"""CREATE VECTOR INDEX {embedding_name} IF NOT EXISTS FOR (n:_Entity_) ON (n.{embedding_name})
                                    OPTIONS {{indexConfig: {{
                                    `vector.dimensions`: {dim},
                                    `vector.similarity_function`: 'cosine'}}
                                    }}"""
        driver.execute_query(query)