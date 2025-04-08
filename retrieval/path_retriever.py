from neo4j import Record, Driver

class PathRetriever:
    QUERIES = {'1hop': """UNWIND $src_names AS srcName
                          MATCH (src {name: srcName})-[r]-(tgt)
                          RETURN labels(src) AS labels1, src.name AS name1, type(r) AS type1, labels(tgt) AS labels2, count(DISTINCT tgt) AS totalCnt""",
               '2hop': """UNWIND $src_names AS srcName
                          MATCH (src1 {name: srcName})-[r1]-(var)-[r2]-(tgt) WHERE tgt <> src1
                          RETURN labels(src1) AS labels1, src1.name AS name1, type(r1) AS type1, labels(var) AS labels2, type(r2) AS type2, labels(tgt) AS labels3, count(DISTINCT tgt) AS totalCnt""",
               '2path': """UNWIND $src_names AS srcName1
                          UNWIND $src_names AS srcName2
                          MATCH (src1 {name: srcName1})-[r1]-(tgt)-[r2]-(src2 {name: srcName2}) WHERE src1 <> src2
                          RETURN labels(src1) AS labels1, src1.name AS name1, type(r1) AS type1, labels(tgt) AS labels2, type(r2) AS type2, labels(src2) AS labels3, src2.name AS name3, count(DISTINCT tgt) AS totalCnt""",
               }

    QUERY_EXTRA_PART_FOR_SUPERVISION = """, size([t IN collect(DISTINCT tgt) WHERE t.nodeId in $tgt_ids| t]) AS correctCnt"""

    def __init__(self, dataset_name):
        match dataset_name:
            case 'prime':
                self.patterns = ['1hop', '2hop', '2path']
                self.specific_target_label = None
            case 'mag':
                self.patterns = ['1hop', '2hop']
                self.specific_target_label = 'Paper'
            case _:
                raise ValueError

    @staticmethod
    def create_query(rec: Record, pattern: str):
        correct_label = lambda labels: list(set(labels).difference({'_Entity_'}))[0]
        match pattern:
            case '1hop':
                return f"MATCH (x1:{correct_label(rec['labels1'])} {{name: \"{rec['name1']}\"}})-[r1:{rec['type1']}]-(x2:{correct_label(rec['labels2'])}) RETURN DISTINCT x2.name AS name"
            case '2hop':
                return f"MATCH (x1:{correct_label(rec['labels1'])} {{name: \"{rec['name1']}\"}})-[r1:{rec['type1']}]-(x2:{correct_label(rec['labels2'])})-[r2:{rec['type2']}]-(x3:{correct_label(rec['labels3'])}) RETURN DISTINCT x3.name AS name"
            case '2path':
                return f"MATCH (x1:{correct_label(rec['labels1'])} {{name: \"{rec['name1']}\"}})-[r1:{rec['type1']}]-(x2:{correct_label(rec['labels2'])})-[r2:{rec['type2']}]-(x3:{correct_label(rec['labels3'])} {{name: \"{rec['name3']}\"}}) RETURN DISTINCT x2.name AS name"
            case _:
                raise ValueError

    def target_has_special_label(self, rec: Record, pattern: str):
        match pattern:
            case '1hop'|'2path':
                return rec['label2'] == self.specific_target_label
            case '2hop':
                return rec['label3'] == self.specific_target_label

    def retrieve_paths(self, driver: Driver, src_names: list[str], tgt_ids: list[str] = None) -> dict:
            supervised = False if tgt_ids is None else True

            paths_data = {'cypher_queries': [], 'hits': [], 'num_results': []}
            for pattern in self.patterns:
                query = PathRetriever.QUERIES[pattern]
                if supervised:
                    query += PathRetriever.QUERY_EXTRA_PART_FOR_SUPERVISION  # to rank best nodes for training llm1 (illegal during testing and inference!)
                for rec in driver.execute_query(query, parameters_={'src_names': src_names, 'tgt_ids': tgt_ids}).records:
                    if self.specific_target_label is not None:
                        if not self.target_has_special_label(rec, pattern): # e.g. only retrieve paper nodes for STaRK-MAG
                            continue
                    paths_data['cypher_queries'].append(PathRetriever.create_query(rec, pattern))
                    paths_data['num_results'].append(rec['totalCnt'])
                    paths_data['hits'].append(rec.get('correctCnt', -1))
            return paths_data