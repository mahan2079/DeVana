import os
import re

def fix_mermaid_labels(content):
    # Regex to find mermaid blocks
    mermaid_blocks = re.findall(r'```mermaid\s*(.*?)\s*```', content, re.DOTALL)
    
    new_content = content
    for block in mermaid_blocks:
        if not ('flowchart' in block or 'graph' in block):
            continue
        
        fixed_block = block
        # Find node definitions: ID[Label], ID(Label), ID{Label}, ID([Label]), etc.
        # This regex matches: 
        # 1. Node ID
        # 2. Opening brackets (one or more of [, (, {)
        # 3. Label (any character except the closing brackets and double quotes, 
        #    containing at least one special character)
        # 4. Closing brackets
        
        # Special characters: ( ) : <br/> + - * / ^ = &
        specials = r'[:\(\)\+\-\*\/\^=&]|\<br\/\>|\\n'
        
        # Pattern to match unquoted labels with special characters
        # Matches: ID[Label with specials] but not ID["Already quoted"]
        pattern = r'(\b[a-zA-Z0-9_-]+)(\[|\(|\{)([^"\]\)\}\n]*?(' + specials + r')[^\]\)\}\n]*?)(\]|\)|\})'
        
        def replace_label(match):
            node_id = match.group(1)
            open_bracket = match.group(2)
            label = match.group(3)
            close_bracket = match.group(5)
            
            # If it's already quoted, don't touch it (though regex tries to avoid it)
            if label.startswith('"') and label.endswith('"'):
                return match.group(0)
            
            return f'{node_id}{open_bracket}"{label}"{close_bracket}'

        # We might need to run it multiple times if there are multiple nodes on one line
        # but re.sub handles multiple matches.
        # However, some nodes have double brackets like ID([Label])
        # Let's handle nested brackets carefully.
        
        # A more specific approach for common mermaid node types
        # ID[Label]
        # ID(Label)
        # ID{Label}
        # ID([Label])
        # ID[[Label]]
        # ID[(Label)]
        # ID((Label))
        
        # Let's use a simpler but broader regex and check the label
        # Match ID then some brackets, then content, then brackets.
        # This handles ID[Label], ID(Label), ID{Label}
        # For nested like ID([Label]), it might match [Label] as the label part.
        
        fixed_block = re.sub(pattern, replace_label, fixed_block)
        
        # Special case for nested brackets like ID([Label])
        # Match ID then ([ then Label then ])
        nested_pattern = r'(\b[a-zA-Z0-9_-]+)(\(\[|\[\[|\[\(|\(\()([^"\]\)\n]*?(' + specials + r')[^\]\)\}\n]*?)(\]\)|\]\]|\)\]|\)\))'
        fixed_block = re.sub(nested_pattern, replace_label, fixed_block)

        if fixed_block != block:
            # Escape backslashes for the replacement string
            safe_block = block.replace('\\', '\\\\')
            safe_fixed_block = fixed_block.replace('\\', '\\\\')
            new_content = new_content.replace(block, fixed_block)
            
    return new_content

def process_files(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    fixed_content = fix_mermaid_labels(content)
                    
                    if fixed_content != content:
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(fixed_content)
                        print(f"Fixed Mermaid syntax in: {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    process_files('.')
