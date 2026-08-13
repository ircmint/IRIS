"""
knowledge_graph.py
--------------------
Builds a multi-relational Knowledge Graph over the whole run:

Node types:
    Video           - the input video/dataset
    Frame           - one sampled frame
    Detection       - a detected feature (marking state / sign / object)
    Violation       - a specific non-compliance finding
    IRCDocument     - IRC35_2015 / IRC67_2022 (kept separate)
    Section         - top-level section of an IRC document
    Clause          - the specific cited clause

Edge types:
    Video -HAS_FRAME-> Frame
    Frame -HAS_DETECTION-> Detection
    Detection -RESULTS_IN-> Violation      (or -COMPLIES_WITH-> for OK signs)
    Violation -CITES-> Clause
    Clause -PART_OF-> Section
    Section -PART_OF-> IRCDocument

The IRC35 and IRC67 sub-graphs are never merged into a single "IRC" node —
each keeps its own IRCDocument node and its own Section/Clause nodes, so the
graph faithfully reflects "each PDF treated separately".

Exports:
    - GraphML (for Gephi / Neo4j import / any graph tool)
    - JSON node/edge list
    - interactive pyvis HTML (self-contained, opens in any browser)
"""

import json
import os
import networkx as nx
from pyvis.network import Network

from config import OUTPUT_DIR

NODE_COLORS = {
    "Video": "#2b6cb0",
    "Frame": "#4299e1",
    "Detection": "#68d391",
    "Violation": "#f56565",
    "Compliant": "#48bb78",
    "IRCDocument": "#805ad5",
    "Section": "#b794f4",
    "Clause": "#d6bcfa",
}


class IRCKnowledgeGraph:
    def __init__(self, video_name="video_dataset"):
        self.g = nx.MultiDiGraph()
        self.video_name = video_name
        self.g.add_node(f"Video::{video_name}", type="Video", label=video_name)
        self._clause_nodes_added = set()

    def _clause_node_id(self, irc_code, clause_id):
        return f"Clause::{irc_code}::{clause_id}"

    def _section_node_id(self, irc_code, section):
        return f"Section::{irc_code}::{section}"

    def _doc_node_id(self, irc_code):
        return f"IRCDocument::{irc_code}"

    def _ensure_clause_chain(self, irc_code, clause):
        """Add Clause -> Section -> IRCDocument chain if not already present."""
        if clause is None:
            return None
        clause_id = clause["clause_id"]
        section = clause_id.split(".")[0]
        doc_id = self._doc_node_id(irc_code)
        sec_id = self._section_node_id(irc_code, section)
        cl_id = self._clause_node_id(irc_code, clause_id)

        if doc_id not in self.g:
            self.g.add_node(doc_id, type="IRCDocument", label=irc_code)
        if sec_id not in self.g:
            self.g.add_node(sec_id, type="Section", label=f"Sec {section}")
            self.g.add_edge(sec_id, doc_id, relation="PART_OF")
        if cl_id not in self.g:
            self.g.add_node(cl_id, type="Clause",
                             label=f"{clause_id}",
                             heading=clause.get("heading", ""),
                             snippet=clause.get("text_snippet", ""),
                             page=clause.get("page"))
            self.g.add_edge(cl_id, sec_id, relation="PART_OF")
        return cl_id

    def add_frame(self, frame_id, timestamp=None):
        frame_node = f"Frame::{frame_id}"
        self.g.add_node(frame_node, type="Frame", label=f"Frame {frame_id}",
                         timestamp=timestamp)
        self.g.add_edge(f"Video::{self.video_name}", frame_node, relation="HAS_FRAME")
        return frame_node

    def add_detection(self, frame_node, det_id, det_label, det_type):
        det_node = f"Detection::{det_id}"
        self.g.add_node(det_node, type="Detection", label=det_label, det_type=det_type)
        self.g.add_edge(frame_node, det_node, relation="HAS_DETECTION")
        return det_node

    def add_violation(self, det_node, violation):
        v_id = f"Violation::{det_node}::{violation['rule']}"
        self.g.add_node(v_id, type="Violation", label=violation["rule"],
                         description=violation["description"],
                         irc_code=violation["irc_code"])
        self.g.add_edge(det_node, v_id, relation="RESULTS_IN")
        cl_id = self._ensure_clause_chain(violation["irc_code"], violation.get("cited_clause"))
        if cl_id:
            self.g.add_edge(v_id, cl_id, relation="CITES")
        return v_id

    def add_compliant(self, det_node, note):
        c_id = f"Compliant::{det_node}::{note['rule']}"
        self.g.add_node(c_id, type="Compliant", label=note["rule"],
                         description=note["description"], irc_code=note["irc_code"])
        self.g.add_edge(det_node, c_id, relation="COMPLIES_WITH")
        return c_id

    def ingest_frame_report(self, frame_id, timestamp, marking_result, sign_detections,
                             object_detections, compliance_report):
        frame_node = self.add_frame(frame_id, timestamp)

        # marking detection node
        m_label = f"marking:{marking_result['colour']}/{marking_result['continuity']}"
        m_det = self.add_detection(frame_node, f"{frame_id}_marking", m_label, "marking")

        # sign detection nodes
        sign_dets = []
        for i, s in enumerate(sign_detections):
            s_label = f"sign:{s['shape']}/{s['condition']}"
            sign_dets.append(self.add_detection(frame_node, f"{frame_id}_sign{i}", s_label, "sign"))

        # object detection nodes
        obj_dets = []
        for i, o in enumerate(object_detections):
            o_label = f"object:{o['class']}"
            obj_dets.append(self.add_detection(frame_node, f"{frame_id}_obj{i}", o_label, "object"))

        # attach violations to the most relevant detection node type
        for v in compliance_report["violations"]:
            if v["rule"] in ("MARKING_ABSENT", "MARKING_FADED", "CONTINUITY_CHECK"):
                target = m_det
            elif v["rule"] in ("SIGN_CONDITION_POOR", "NO_SIGN_VISIBLE"):
                target = sign_dets[0] if sign_dets else m_det
            else:
                target = obj_dets[0] if obj_dets else m_det
            self.add_violation(target, v)

        for c in compliance_report.get("compliant_signs", []):
            target = sign_dets[0] if sign_dets else m_det
            self.add_compliant(target, c)

        return frame_node

    # ------------------------------------------------------------------
    def export_graphml(self, path=None):
        path = path or os.path.join(OUTPUT_DIR, "irc_knowledge_graph.graphml")
        g2 = nx.MultiDiGraph()
        for n, d in self.g.nodes(data=True):
            clean = {k: ("" if v is None else v) for k, v in d.items()}
            g2.add_node(n, **clean)
        for u, v, d in self.g.edges(data=True):
            g2.add_edge(u, v, **d)
        nx.write_graphml(g2, path)
        return path

    def export_json(self, path=None):
        path = path or os.path.join(OUTPUT_DIR, "irc_knowledge_graph.json")
        data = nx.node_link_data(self.g, edges="edges")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str, ensure_ascii=False)        
        return path

    def export_html(self, path=None):
        path = path or os.path.join(OUTPUT_DIR, "irc_knowledge_graph.html")
        net = Network(height="850px", width="100%", directed=True,
                      bgcolor="#111827", font_color="white", notebook=False,
                      cdn_resources="in_line")
        for n, d in self.g.nodes(data=True):
            ntype = d.get("type", "Detection")
            color = NODE_COLORS.get(ntype, "#a0aec0")
            title_bits = [f"{k}: {v}" for k, v in d.items()]
            size = {"Video": 40, "IRCDocument": 34, "Section": 22,
                    "Frame": 18, "Clause": 16}.get(ntype, 12)
            net.add_node(n, label=d.get("label", n), title="\n".join(title_bits),
                         color=color, size=size)
        for u, v, d in self.g.edges(data=True):
            net.add_edge(u, v, title=d.get("relation", ""), arrows="to")

        net.set_options("""
        {
          "physics": {"solver": "forceAtlas2Based",
            "forceAtlas2Based": {"gravitationalConstant": -60, "springLength": 100},
            "minVelocity": 0.75},
          "edges": {"color": {"color": "#4a5568"}, "smooth": false},
          "interaction": {"hover": true, "tooltipDelay": 100}
        }
        """)
        html = net.generate_html(notebook=False)

        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        return path

    def summary_stats(self):
        by_type = {}
        for _, d in self.g.nodes(data=True):
            by_type[d.get("type", "?")] = by_type.get(d.get("type", "?"), 0) + 1
        return {
            "total_nodes": self.g.number_of_nodes(),
            "total_edges": self.g.number_of_edges(),
            "nodes_by_type": by_type,
        }