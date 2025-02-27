import re

def cypher2path(cypher_query: str) -> list[tuple[str, str, str, str]]:
    #path = re.findall(r"(?:\(|-\[)(x|r)(\d):([^ \)\]]+)(?: \{name: \"(.+)\"\})?(?:\)|\]-)", cypher_query)
    path = re.findall(r"(?:\(|-\[)(x|r)(\d):([^ \)\]]+)(?: \{name: \"([^\"]+)\"\})?(?:\)|\]-)", cypher_query)
    return path

def block2cypher(x_r: str, num: str, label_or_type: str, name: str) -> str:
    if x_r == 'x':
        prop_string = f" {{name: \"{name}\"}}" if name != '' else ""
        return f"(x{num}:{label_or_type}{prop_string})"
    elif x_r == 'r':
        return f"-[r{num}:{label_or_type}]-"

def path2cypher(path: list[tuple[str, str, str, str]]) -> str:
    query = "MATCH "
    for x_r, num, labelOrType, name in path:
        if x_r == 'x' or x_r == 'r':
            query += block2cypher(x_r, num, labelOrType, name)
        elif x_r == '':
            query += f" RETURN x{num}.name as name"
    return query