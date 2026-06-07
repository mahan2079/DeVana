import os
import re

def quote_mermaid_nodes(mermaid_block):
    lines = mermaid_block.split('\n')
    new_lines = []
    
    # We will process each line
    for line in lines:
        if line.strip() in ['```mermaid', '```'] or line.strip().startswith(('flowchart', 'graph', 'subgraph', 'end', '%%', 'style ', 'linkStyle ', 'classDef ', 'class ')):
            new_lines.append(line)
            continue
            
        new_line = line
        
        # Match node definitions. They usually look like ID[Content] or ID(Content).
        # We need a regex that finds ID, an open bracket, content, and a closing bracket.
        # But we must be careful with nested brackets.
        
        # Patterns for: [[ ]], [( )], (( )), {{ }}, [ ], ( ), { }, > ]
        patterns = [
            r'(\b[a-zA-Z0-9_-]+)(\[\[)(.*?)(\]\])',
            r'(\b[a-zA-Z0-9_-]+)(\[\()(.*?)(\)\])',
            r'(\b[a-zA-Z0-9_-]+)(\(\()(.*?)(\)\))',
            r'(\b[a-zA-Z0-9_-]+)(\{\{)(.*?)(\}\})',
            r'(\b[a-zA-Z0-9_-]+)(\[)(.*?)(\])',
            r'(\b[a-zA-Z0-9_-]+)(\()(.*?)(\))',
            r'(\b[a-zA-Z0-9_-]+)(\{)(.*?)(\})',
            r'(\b[a-zA-Z0-9_-]+)(\>)(.*?)(\])',
        ]
        
        for pat in patterns:
            def safe_replacer(m):
                nid = m.group(1)
                ob = m.group(2)
                content = m.group(3)
                cb = m.group(4)
                
                # Exclude strings with arrow patterns to avoid cross-node matches
                if '-->' in content or '---' in content or '==>' in content:
                    return m.group(0)
                    
                # If already quoted properly, do nothing
                if content.startswith('"') and content.endswith('"'):
                    return m.group(0)
                    
                # If it's a subgraph name or something similar, skip it (subgraph handled above, but just in case)
                if nid == 'subgraph':
                    return m.group(0)
                
                clean_content = content.replace('"', '\\"')
                return f'{nid}{ob}"{clean_content}"{cb}'

            # Apply pattern. Run it multiple times to catch multiple nodes on the same line.
            new_line = re.sub(pat, safe_replacer, new_line)
            
        new_lines.append(new_line)
        
    return '\n'.join(new_lines)

def append_pseudo_code(content, file_path):
    mermaid_pattern = re.compile(r'```mermaid\n(.*?)\n```', re.DOTALL)
    matches = list(mermaid_pattern.finditer(content))
    if not matches:
        return content
        
    new_content = content
    offset = 0
    
    for match in matches:
        original_block = match.group(0)
        
        if not re.search(r'\b(flowchart|graph)\b', original_block, re.IGNORECASE):
            continue
            
        fixed_block = quote_mermaid_nodes(original_block)
        
        start_idx = match.end() + offset
        lookahead = new_content[start_idx:start_idx+200]
        has_pseudo = bool(re.search(r'#+\s*(Flowchart\s*)?Pseudo-?code', lookahead, re.IGNORECASE))
        
        replacement = fixed_block
        if not has_pseudo:
            # Generate pseudo-code
            nodes = []
            # Extract text from "..." inside brackets
            # First, match quoted text inside nodes
            for line in fixed_block.split('\n'):
                # match NodeID["Text"] or similar
                # Just find everything inside quotes for simplicity
                quoted_texts = re.findall(r'"([^"]*?)"', line)
                for txt in quoted_texts:
                    clean_lbl = txt.replace('<br/>', ' ').replace('\\n', ' ').strip()
                    if clean_lbl and clean_lbl not in nodes:
                        nodes.append(clean_lbl)
            
            pseudo = "\n\n#### Pseudo-code\n```text\nBEGIN\n"
            for lbl in nodes:
                pseudo += f"  EXECUTE {lbl}\n"
            pseudo += "END\n```"
            replacement += pseudo
            
        if replacement != original_block:
            new_content = new_content[:match.start()+offset] + replacement + new_content[start_idx:]
            offset += len(replacement) - len(original_block)
            
    return new_content

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    new_content = append_pseudo_code(content, file_path)
    
    if new_content != content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated: {file_path}")

for root, dirs, files in os.walk('Documents'):
    for file in files:
        if file.endswith('.md'):
            process_file(os.path.join(root, file))
