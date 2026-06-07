import os
import ast
from pathlib import Path

MIXINS_DIR = Path("codes/gui")
MAIN_WINDOW_DIR = MIXINS_DIR / "main_window"
DOCS_DIR = Path("Documents/GUI_Guild/Mixins")

docs_dir = DOCS_DIR
docs_dir.mkdir(parents=True, exist_ok=True)

def generate_doc_for_mixin(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    tree = ast.parse(content)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    
    if not classes:
        return None

    doc_content = ""
    for cls in classes:
        cls_name = cls.name
        methods = [m for m in cls.body if isinstance(m, ast.FunctionDef)]
        
        doc_content += f"# Reference: {cls_name}\n\n"
        doc_content += f"## Overview\n"
        doc_content += f"This document provides reference documentation for the `{cls_name}` in `{file_path.name}`. It is structured according to the Diátaxis framework.\n\n"
        
        doc_content += f"## Flowchart\n"
        doc_content += f"```mermaid\n"
        doc_content += f"flowchart TD\n"
        doc_content += f'    Start(["User Action"]) --> Init["Initialize {cls_name}"]\n'
        for i, m in enumerate(methods[:5]):
            doc_content += f'    Init --> M{i}["{m.name}"]\n'
        doc_content += f"```\n\n"
        
        doc_content += f"### Flowchart Pseudo-code\n"
        doc_content += f"```text\n"
        doc_content += f"BEGIN {cls_name} Initialization\n"
        for m in methods[:5]:
            doc_content += f"  CALL {m.name}\n"
        doc_content += f"END\n"
        doc_content += f"```\n\n"
        
        doc_content += f"## Reference\n"
        for m in methods:
            doc_content += f"### `{m.name}`\n"
            doc_content += f"**Description:** Executes logic for `{m.name}`.\n\n"
            
    return doc_content

mixin_files = list(MIXINS_DIR.glob("*_mixin.py")) + list(MAIN_WINDOW_DIR.glob("*_mixin.py"))

for file_path in mixin_files:
    # check if already documented
    base_name = file_path.stem.replace("_mixin", "").capitalize()
    
    # special cases
    mapping = {
        "Nsga2": "NSGA2", "Pso": "PSO", "Ga": "GA", "Cmaes": "CMAES",
        "Adavea": "AdaVEA", "Frf": "FRF", "Rl": "RL", "Pinn": "PINN",
        "Moga": "MOGA", "De": "DE", "Sa": "SA", "Sobol": "Sobol",
        "Omega_sensitivity": "Omega_Sensitivity", "Extra_opt": "Extra_Opt",
        "Api_key": "API_Key", "Ai_assistant": "AI_Assistant", "Stochastic": "Stochastic",
        "Input": "Input", "Beam": "Beam", "Microchip": "Microchip"
    }
    
    doc_name = mapping.get(base_name, base_name)
    doc_path = docs_dir / f"{doc_name}.md"
    
    if not doc_path.exists():
        doc = generate_doc_for_mixin(file_path)
        if doc:
            with open(doc_path, "w", encoding="utf-8") as f:
                f.write(doc)
            print(f"Generated: {doc_path}")
