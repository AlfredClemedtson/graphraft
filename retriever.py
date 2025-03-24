from neo4j import Driver


def extract_from_query(query: str):
    tgt = query.split("RETURN ")[1].split(".")[0]  # ... RETURN tgtVarName.name,...  --> tgtVarName  #Replace with regex
    tgt_label = query.split(tgt)[1].split(":")[1].split(")")[0]  # MATCH...(tgtVarName:tgtLabel)...  #Replace with regex
    pattern = query.split("MATCH ")[-1].split("RETURN")[0]  # Replace
    return tgt, tgt_label, pattern

def query_to_text_pattern(query, rec, key):
    tgt, tgt_label, pattern = extract_from_query(query)
    pattern = pattern.replace(f"({tgt}:{tgt_label})", f"({tgt}:{tgt_label} {{{key}: \"{rec[key]}\"}})") # Replace
    return pattern

VECTOR_SIMILARITY_QUERY = """CALL db.index.vector.queryNodes($vectorIndex, $ef, $questionEmbedding) 
                             YIELD node AS node, score
                             WHERE NOT node.nodeId IN $foundNodeIds
                             RETURN score AS similarity, node.nodeId AS nodeId"""

ANSWER_NAMES_QUERY = """UNWIND $answerIds AS nodeId
                        MATCH (x:_Entity_ {nodeId: nodeId})
                        RETURN x.name as name"""

class Retriever:
    def __init__(self, node_properties: list[str], sorting_index="nameEmbedding", vector_index="abstractEmbedding",
                 pattern_rate=1, ef=10_000, count_tokens=False, max_nodes=None, max_tokens=None, formatter=None,
                 tokenizer=None):
        self.node_properties = node_properties
        self.sorting_index = sorting_index
        self.vector_index = vector_index
        self.pattern_rate = pattern_rate
        self.ef = ef

        self.count_tokens = count_tokens
        if count_tokens:
            if max_tokens is None or tokenizer is None:
                raise ValueError("max_tokens or tokenizer cannot be None if counting tokens.")
            else:
                self.max_tokens, self.tokenizer = max_tokens, tokenizer
            if formatter is None:
                format_node_data = lambda x: '\n'.join([f"{key}: {value}" for key, value in x.items() if value is not None and key not in {'nodeId', 'similarity'}])
                self.formatter = lambda xs:  '\n\n'.join([format_node_data(x) for x in xs])
            else:
                self.formatter = formatter
        else:  # count nodes
            if max_nodes is None:
                raise ValueError("max_nodes cannot be None if not counting tokens.")
            self.max_nodes = max_nodes


    def modify_query(self, query: str, vector_sim=False):
        if vector_sim:
            for prop in self.node_properties:
                query += f", node.{prop} AS {prop}"
            return query
        else:
            tgt, _, pattern = extract_from_query(query)
            query = f"MATCH {pattern} RETURN {tgt}.nodeId as nodeId"
            for prop in self.node_properties:
                query += f", {tgt}.{prop} AS {prop}"
            query += f", vector.similarity.cosine({tgt}.{self.sorting_index}, $questionEmbedding) AS similarity ORDER BY similarity DESC"
        return query


    def stop_retrieval(self, retrieved_nodes: list[dict], rate) -> bool:
        if not self.count_tokens:
            return len(retrieved_nodes) >= rate * self.max_nodes
        else:
            formatted_node_data = self.formatter(retrieved_nodes)
            tokenized_node_data = self.tokenizer(formatted_node_data)
            return len(tokenized_node_data) >= rate*self.max_tokens


    def retrieve_data(self, driver:Driver, cypher_queries: list[str], q_emb=None):
        #If q_emb is not given, generate it!
        retrieved_data = []
        for cypher_query in cypher_queries:
            cypher_query = self.modify_query(cypher_query)
            try:
                with driver.session() as session:
                    for rec in session.run(cypher_query, parameters={'questionEmbedding': q_emb}):
                        rec = dict(rec) | {'pattern': query_to_text_pattern(cypher_query, rec, key='name')}
                        retrieved_data.append(rec)
                        if self.stop_retrieval(retrieved_data, rate=self.pattern_rate):
                            break
                    else:
                        continue
                    break #To break outer if inner is broken not finished
            except Exception as e:
                print(e)
        with driver.session() as session:
            cypher_query = self.modify_query(VECTOR_SIMILARITY_QUERY, vector_sim=True)
            for rec in session.run(cypher_query,
                                   parameters={'vectorIndex': self.vector_index, 'ef': self.ef,
                                               'questionEmbedding': q_emb,
                                               'foundNodeIds': [data['nodeId'] for data in retrieved_data]}):
                rec = dict(rec) | {'pattern' : "No pattern"}
                retrieved_data.append(rec)
                if self.stop_retrieval(retrieved_data, rate=1):
                    break
        return retrieved_data
    
    def retrieve_answer_names(self, driver: Driver, answer_ids: list[int]):
        res = driver.execute_query(ANSWER_NAMES_QUERY, parameters_={'answerIds': answer_ids})
        names = [rec['name'] for rec in res.records]
        return names