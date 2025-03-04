import networkx as nx
import json
import spacy
import os
import fitz  

class KnowledgeGraph:
    def __init__(self, graph_path="data/graph_kg.json", pdf_folder="data/documents"):
        self.graph = nx.Graph()
        self.graph_path = graph_path
        self.pdf_folder = pdf_folder
        self.nlp = spacy.load("en_core_web_sm")  
        self.load_graph()

    def extract_text_from_pdf(self, pdf_path):
        """ Extract text from a given PDF file """
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        return text.strip()

    def extract_entities_relations(self, text):
        """Extracts named entities and their relations using dependency parsing."""
        doc = self.nlp(text)
        entities = {ent.text: ent.label_ for ent in doc.ents}  # Extract entities with types
        relations = []

        for token in doc:
            # Look for verbs that indicate relationships
            if token.pos_ == "VERB":
                subject = [child.text for child in token.lefts if child.dep_ in ("nsubj", "nsubjpass")]
                object_ = [child.text for child in token.rights if child.dep_ in ("dobj", "pobj", "attr")]

                if subject and object_:
                    relations.append((subject[0], token.text, object_[0])) 

        return list(entities.keys()), relations 


    def process_pdfs(self):
        """Processes all PDFs in the data folder and builds the knowledge graph."""
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith(".pdf")]
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, pdf_file)
            print(f"Processing {pdf_file}...")
            
            text = self.extract_text_from_pdf(pdf_path)
            entities, relations = self.extract_entities_relations(text)

            # Add entities (nodes) to the graph
            for entity in entities:
                self.graph.add_node(entity)

            # Add relationships (edges with labels)
            for ent1, relation, ent2 in relations:
                self.graph.add_edge(ent1, ent2, relation=relation)


    def query_graph(self, entity):
        """ Retrieves related concepts from the knowledge graph """
        if entity in self.graph:
            return list(nx.neighbors(self.graph, entity))
        return []

    def save_graph(self):
        """ Saves the graph to a JSON file """
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, "w") as f:
            json.dump(data, f)

    def load_graph(self):
        """ Loads the graph from a JSON file if it exists """
        try:
            with open(self.graph_path, "r") as f:
                data = json.load(f)
                self.graph = nx.node_link_graph(data)
        except FileNotFoundError:
            print("No existing graph found. Creating a new one.")

if __name__ == "__main__":
    kg = KnowledgeGraph()
    kg.process_pdfs()
    kg.save_graph()
    print("Graph RAG built successfully!")
