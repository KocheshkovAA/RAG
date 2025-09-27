from neo4j import GraphDatabase
from app.config import NEO4J_USER, NEO4J_PASSWORD, NEO4J_URI
from itertools import combinations

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

from neo4j import GraphDatabase
from app.config import NEO4J_USER, NEO4J_PASSWORD, NEO4J_URI

driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_node_info(node_title: str, detailed: bool = False):
    with driver.session() as session:
        if detailed:
            query = """
            MATCH (n {title: $title})
            OPTIONAL MATCH (n)-[out_rel]->(out_node)
              WHERE NOT type(out_rel) IN ['ССЫЛКА', 'ПРИНАДЛЕЖНОСТЬ', 'УЧАСТНИК', 'ПРЕДЫДУЩАЯ', 'СЛЕДУЮЩАЯ']
            OPTIONAL MATCH (in_node)-[in_rel]->(n)
              WHERE NOT type(in_rel) IN ['ССЫЛКА', 'ПРИНАДЛЕЖНОСТЬ', 'УЧАСТНИК', 'ПРЕДЫДУЩАЯ', 'СЛЕДУЮЩАЯ']
            RETURN n.title AS title,
                   n.first_paragraph AS text,
                   labels(n) AS labels,
                   collect(DISTINCT {rel: type(out_rel), target: out_node.title}) AS outgoing,
                   collect(DISTINCT {rel: type(in_rel), source: in_node.title}) AS incoming
            """
        else:
            query = """
            MATCH (n {title: $title})
            RETURN n.title AS title,
                   n.first_paragraph AS text,
                   labels(n) AS labels
            """

        record = session.run(query, title=node_title).single()
        if not record:
            return None

        if not detailed:
            output = f"=== {record['title']}"
            if record.get("labels"):
                output += f" [{', '.join(record['labels'])}]"
            output += " ===\n"
            if record["text"]:
                output += f"Описание: {record['text']}\n"
            return output

        outgoing = [r for r in record["outgoing"] if r.get("target")]
        incoming = [r for r in record["incoming"] if r.get("source")]

        output = f"=== {record['title']} [{', '.join(record['labels'])}] ===\n"
        if record["text"]:
            output += f"Описание: {record['text']}\n\n"

        if outgoing:
            output += "Исходящие связи:\n"
            for rel in outgoing:
                output += f"  - {rel['rel']}: {rel['target']}\n"

        if incoming:
            output += "\nВходящие связи:\n"
            for rel in incoming:
                output += f"  - {rel['rel']}: {rel['source']}\n"

        return output


def calculate_graph_metrics(nodes: list, max_length=5):
    """
    Возвращает:
    1) node_scores: {узел: графовый скор}
    2) paths_between_nodes: {(node1, node2): путь с типами связей}
    3) intermediate_nodes: список промежуточных узлов, которых нет в nodes
    """
    node_scores = {node: 0.0 for node in nodes}
    paths_between_nodes = {}
    intermediate_nodes = set()

    with driver.session() as session:
        for node1, node2 in combinations(nodes, 2):
            query = f"""
            MATCH p=(a {{title: $node1}})-[rels*..{max_length}]-(b {{title: $node2}})
            WHERE all(r IN rels 
                      WHERE type(r) <> 'РАСА' 
                        AND type(r) <> 'СТАТУС' 
                        AND type(r) <> 'ССЫЛКА'
                        AND type(r) <> 'ПРЕДСТАВЛЯЕТ'
                        AND type(r) <> 'ПОТЕРИ'
                        AND type(r) <> 'ВОЙСКА'
                        AND type(r) <> 'ПОГИБ'
                        AND type(r) <> 'ДАТА'
                        AND type(r) <> 'СЕГМЕНТУМ'
                        AND type(r) <> 'СЕКТОР'
                        AND type(r) <> 'ЖАНР'
                        AND type(r) <> 'ПРЕДЫДУЩАЯ'
                        AND type(r) <> 'ИЗДАТЕЛЬ'
                        AND type(r) <> 'СЛЕДУЮЩАЯ'
                        AND type(r) <> 'ПРИНАДЛЕЖНОСТЬ'
                        AND type(r) <> 'ЯВЛЯЮТСЯ_НАСЛЕДНИКАМИ')
              AND NONE(n IN nodes(p)[1..-1] WHERE 
                    ANY(l IN labels(n) WHERE l IN ['Персонажи_', 'Организации_Империума'])
                    OR n.title IN ['Неизвестно', 'Неизвестен']
                )
            RETURN [n IN nodes(p) | n.title] AS path,
                   [r IN relationships(p) | type(r)] AS rels,
                   length(p) AS path_length
            ORDER BY path_length ASC
            LIMIT 1
            """
            record = session.run(query, node1=node1, node2=node2).single()
            if record and record["path_length"] is not None:
                dist = record["path_length"]
                score = 1 / (1 + dist)
                node_scores[node1] += score
                node_scores[node2] += score

                path_nodes = record["path"]
                path_rels = record["rels"]

                # Формируем путь с типами связей
                path_with_rels = []
                for i in range(len(path_nodes) - 1):
                    path_with_rels.append(path_nodes[i])
                    path_with_rels.append(f"-[{path_rels[i]}]->")
                path_with_rels.append(path_nodes[-1])

                paths_between_nodes[(node1, node2)] = path_with_rels
                paths_between_nodes[(node2, node1)] = path_with_rels  # симметрично

                # Добавляем промежуточные узлы
                for n in path_nodes[1:-1]:
                    if n not in nodes:
                        intermediate_nodes.add(n)
            else:
                # путь не найден
                node_scores[node1] += 0
                node_scores[node2] += 0

    return node_scores, paths_between_nodes, list(intermediate_nodes)
