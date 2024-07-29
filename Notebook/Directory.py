import os
from graphviz import Digraph
os.getcwd()
def add_nodes_edges(dot, path, parent=None):
    for root, dirs, files in os.walk(path):
        for d in dirs:
            dot.node(d)
            if parent:
                dot.edge(parent, d)
            add_nodes_edges(dot, os.path.join(root, d), d)
        for f in files:
            dot.node(f)
            if parent:
                dot.edge(parent, f)
        break

def create_directory_graph(path):
    dot = Digraph(comment='Directory Structure')
    root = os.path.basename(path)
    dot.node(root, root)
    add_nodes_edges(dot, path, root)
    dot.render('directory-graph', format='png')

create_directory_graph('/Users/girimanoharv/Documents/Social-Media-Sentiment-Analysis')
