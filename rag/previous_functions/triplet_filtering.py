import os
import networkx as nx
import pickle
import spacy
nlp = spacy.load("es_core_news_lg") 


KG_PATH  = "/data/users/pfont/graph/online_knowledge_graph_tests.pkl"


# def load_knowledge_graph(filepath):
#     if os.path.exists(filepath):
#         try:
#             with open(filepath, 'rb') as f:
#                 graph = pickle.load(f)
#             print(f"Knowledge graph loaded from {filepath}. Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}")
#             return graph
#         except Exception as e:
#             print(f"Error loading knowledge graph: {e}. Starting with an empty graph.")
#             return nx.MultiDiGraph()
#     else:
#         print("No saved knowledge graph found. Starting with an empty graph.")
#         return nx.MultiDiGraph()
# graph = load_knowledge_graph(KG_PATH)


# nodes = graph.nodes()
# print(nodes)
# new_nodes =[] 
# for node in nodes:
#     doc = nlp(str(node))
#     new_nodes.append(doc)
# new_nodes

# print(new_nodes)


string = "La sentencia del Consejo de Guerra Ordinario de Plaza, presidido por el Teniente Coronel D. Rodrigo Torrent, condenó a Francisco Iacasta Catalán a dos años y seis meses de reclusión menor por el delito de auxilio a la Rebelión. Esta decisión se basó en el artículo 240 del Código de Justicia Militar de 1940, sin circunstancias modificativas de responsabilidad criminal. Aunque la defensa solicitó la absolución o una pena menor, el fallo fue confirmado por los vocales del consejo. Se menciona un voto particular que discrepa, proponiendo una pena de un año de reclusión menor."
doc = nlp(string)

print([(w.text, w.pos_) for w in doc])
print(type(doc))

person_names = [ent.text for ent in doc.ents ]
print("Nombres de personas encontrados:", person_names)
