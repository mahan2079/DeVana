import os
from pathlib import Path

def create_master_markdown():
    output_file = "DeVana_Master_Documentation.md"
    
    # Define the intended order of sections for the master document
    priority_order = [
        "Documents/INDEX.md",
        "Documents/SOVEREIGN_LEDGER.md",
        "Documents/Optimization_Guild/README.md",
        "Documents/GUI_Guild/README.md",
        "Documents/API_Library_Guild/README.md",
        "Documents/Physics_Guild/README.md",
    ]
    
    # Folders to crawl for the remaining content
    guild_folders = [
        "Documents/Optimization_Guild",
        "Documents/GUI_Guild",
        "Documents/API_Library_Guild",
        "Documents/Physics_Guild",
        "Documents/Algorithms",
        "Documents/Analysis",
        "Documents/CoreComponents",
    ]

    # Tracking included files to prevent duplicates
    included_files = set()
    
    master_content = [
        "# 🌌 DeVana: Sovereign Master Documentation",
        "**Generated on:** June 7, 2026",
        "**Architect:** Dolores (Sovereign Omniscient Architect)",
        "\n---\n",
        "## 📑 Table of Contents (Virtual)",
        "This document is a unified synthesis of the entire DeVana architectural lake. It includes Optimization, GUI, API, Library, and Physics domains.",
        "\n---\n"
    ]

    def add_file_to_master(file_path):
        p = Path(file_path)
        if not p.exists() or str(p) in included_files:
            return
        
        print(f"Synthesizing: {file_path}")
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Add a clear section header for the file
            header = f"\n\n{'='*80}\n"
            header += f"## SOURCE: {file_path}\n"
            header += f"{'='*80}\n\n"
            
            master_content.append(header)
            master_content.append(content)
            included_files.add(str(p))
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # 1. Process Priority Files
    for f in priority_order:
        add_file_to_master(f)

    # 2. Process Folders
    for folder in guild_folders:
        folder_path = Path(folder)
        if not folder_path.exists():
            continue
            
        # Sort files to maintain some order within folders
        for root, dirs, files in os.walk(folder_path):
            for file in sorted(files):
                if file.endswith(".md"):
                    full_path = os.path.join(root, file)
                    add_file_to_master(full_path)

    # 3. Process anything else in Documents/ that might have been missed
    for root, dirs, files in os.walk("Documents"):
        for file in sorted(files):
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                add_file_to_master(full_path)

    # Write the master file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(master_content))
    
    print(f"\n✅ SUCCESS: Integrated {len(included_files)} documents into {output_file}")

if __name__ == "__main__":
    create_master_markdown()
