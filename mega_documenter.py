import ast
import os
import sys

class MegaDocumenter:
    def __init__(self, file_path, target_dir):
        self.file_path = file_path
        self.target_dir = target_dir
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            self.source = f.read()
        self.tree = ast.parse(self.source)
        self.filename = os.path.basename(file_path)
        self.classname = self._get_class_name()
        self.methods = self._get_methods()

    def _get_class_name(self):
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                return node.name
        return self.filename.replace('.py', '')

    def _get_methods(self):
        methods = []
        for node in self.tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        methods.append(item)
            elif isinstance(node, ast.FunctionDef):
                methods.append(node)
        return methods

    def _summarize_logic(self, node):
        steps = []
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            if isinstance(stmt, ast.Assign):
                steps.append(f"Assigns {ast.unparse(stmt.targets[0])}")
            elif isinstance(stmt, ast.Call):
                steps.append(f"Calls {ast.unparse(stmt.func).split('.')[-1]}")
            elif isinstance(stmt, ast.If):
                steps.append(f"Conditional: {ast.unparse(stmt.test)[:30]}")
            elif isinstance(stmt, ast.For):
                steps.append(f"Loops over {ast.unparse(stmt.iter)[:30]}")
            elif isinstance(stmt, ast.Return):
                steps.append("Returns result")
        
        if not steps:
            return "Simple function logic."
        return "; ".join(steps[:5]) + ("..." if len(steps) > 5 else "")

    def _generate_flowchart(self, method_node):
        if not method_node:
            return "N/A", "N/A"
        
        nodes = []
        nodes.append(f'Start["Start: {method_node.name}"]')
        
        stmt_count = 0
        for stmt in method_node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                continue
            
            desc = ""
            if isinstance(stmt, ast.Assign):
                desc = f"Initialize {ast.unparse(stmt.targets[0])[:20]}"
            elif isinstance(stmt, ast.Call):
                desc = f"Invoke {ast.unparse(stmt.func).split('.')[-1][:20]}"
            elif isinstance(stmt, ast.If):
                desc = f"Validate {ast.unparse(stmt.test)[:20]}"
            elif isinstance(stmt, ast.For):
                desc = f"Iterate {ast.unparse(stmt.iter)[:20]}"
            elif isinstance(stmt, ast.Try):
                desc = "Error Handling Block"
            elif isinstance(stmt, ast.With):
                desc = "Resource Context"
            
            if desc:
                nodes.append(f'Step{stmt_count}["{desc}"]')
                stmt_count += 1
            
            if stmt_count >= 8:
                break
        
        nodes.append(f'End["End: {method_node.name}"]')
        
        mermaid = "```mermaid\ngraph TD\n"
        for i in range(len(nodes) - 1):
            mermaid += f"    {nodes[i]} --> {nodes[i+1]}\n"
        mermaid += "```"

        pseudo = "```python\n"
        pseudo += f"FUNCTION {method_node.name}({', '.join([arg.arg for arg in method_node.args.args])}):\n"
        for node_str in nodes[1:-1]:
            label = node_str.split('["')[1].split('"]')[0]
            pseudo += f"    DO \"{label}\"\n"
        pseudo += "END FUNCTION\n```"
        
        return mermaid, pseudo

    def document(self):
        doc = f"# {self.classname}\n\n"
        
        # Purpose
        class_doc = ast.get_docstring(self.tree)
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef):
                class_doc = ast.get_docstring(node)
                break
        
        doc += "## Purpose\n"
        doc += (class_doc if class_doc else f"Core implementation of {self.classname} logic.") + "\n\n"
        
        # Significant Method Flowchart
        sig_method = None
        for m in self.methods:
            if m.name in ['run', 'create_page', 'setup_ui'] or m.name.startswith('init_') or m.name.startswith('create_'):
                sig_method = m
                break
        if not sig_method and self.methods:
            sig_method = self.methods[0]
            
        if sig_method:
            doc += f"## Internal Logic Flow: `{sig_method.name}`\n"
            mermaid, pseudo = self._generate_flowchart(sig_method)
            doc += mermaid + "\n\n"
            doc += "### Flowchart Pseudo-code\n"
            doc += pseudo + "\n\n"

        # Function Documentation
        doc += "## Methods & Functions\n\n"
        for m in self.methods:
            args = [arg.arg for arg in m.args.args]
            ret = ast.unparse(m.returns) if m.returns else "None"
            summary = self._summarize_logic(m)
            
            doc += f"### `{m.name}`\n"
            doc += f"- **Arguments**: `{', '.join(args)}`\n"
            doc += f"- **Returns**: `{ret}`\n"
            doc += f"- **Logic**: {summary}\n\n"

        target_path = os.path.join(self.target_dir, self.filename.replace('.py', '.md'))
        os.makedirs(self.target_dir, exist_ok=True)
        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(doc)
        print(f"Generated: {target_path}")

def main():
    targets = [
        ('codes/gui/', 'Documents/GUI_Guild/Mixins/'),
        ('codes/gui/main_window/', 'Documents/GUI_Guild/Mixins/'),
        ('codes/workers/', 'Documents/Optimization_Guild/Workers/')
    ]
    
    for src_dir, dest_dir in targets:
        if not os.path.exists(src_dir):
            continue
        for f in os.listdir(src_dir):
            if f.endswith('.py') and f != '__init__.py':
                file_path = os.path.join(src_dir, f)
                doc_gen = MegaDocumenter(file_path, dest_dir)
                doc_gen.document()

if __name__ == "__main__":
    main()
