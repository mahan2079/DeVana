import os
import re
import requests
import base64
from pathlib import Path
from md2pdf.core import md2pdf

# Kroki API for Mermaid to PNG conversion
KROKI_URL = "https://kroki.io/mermaid/png/"

def mermaid_to_image_kroki(mermaid_code):
    """Converts Mermaid code to a PNG image using the Kroki API."""
    try:
        # Kroki expects the mermaid code to be compressed or just posted
        # We'll use the simple GET interface with base64 encoding if needed, 
        # but Kroki also supports POST with the raw text.
        response = requests.post(KROKI_URL, data=mermaid_code.encode('utf-8'), timeout=10)
        if response.status_code == 200:
            return response.content
    except Exception as e:
        print(f"Error calling Kroki: {e}")
    return None

def process_markdown_with_images(content, img_dir):
    """Replaces Mermaid blocks with local image links."""
    mermaid_blocks = re.findall(r"```mermaid\n(.*?)\n```", content, re.DOTALL)
    
    for i, block in enumerate(mermaid_blocks):
        img_data = mermaid_to_image_kroki(block)
        if img_data:
            img_filename = f"mermaid_{hash(block) & 0xffffffff}.png"
            img_path = img_dir / img_filename
            with open(img_path, "wb") as f:
                f.write(img_data)
            
            # Replace the block with a Markdown image link
            # Note: We keep the pseudo-code as well since it's already there in the files
            content = content.replace(f"```mermaid\n{block}\n```", f"![Flowchart]({img_path.absolute()})")
    
    return content

def generate_pro_pdf():
    output_pdf = "DeVana_Professional_Documentation.pdf"
    img_dir = Path("temp_images")
    img_dir.mkdir(exist_ok=True)
    
    # Priority Order
    priority_files = [
        "Documents/INDEX.md",
        "Documents/Optimization_Guild/README.md",
        "Documents/GUI_Guild/README.md",
        "Documents/API_Library_Guild/README.md",
        "Documents/Physics_Guild/README.md",
    ]
    
    # Discovery
    all_files = []
    for root, dirs, files in os.walk("Documents"):
        for file in files:
            if file.endswith(".md"):
                all_files.append(os.path.join(root, file))
    
    # Sort and Filter
    final_list = [f for f in priority_files if os.path.exists(f)]
    remaining = [f for f in all_files if f not in final_list]
    final_list.extend(sorted(remaining))
    
    # Combine into one massive MD
    combined_md = "# DeVana Sovereign Documentation\n\nGenerated on: June 7, 2026\n\n---\n\n"
    
    for file_path in final_list:
        print(f"Processing {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            file_content = f.read()
        
        # Add file header
        combined_md += f"\n\n---\n\n## SOURCE: {file_path}\n\n"
        combined_md += process_markdown_with_images(file_content, img_dir)
    
    # Save combined MD for rendering
    temp_md_path = Path("PRO_DOC_TEMP.md")
    with open(temp_md_path, "w", encoding="utf-8") as f:
        f.write(combined_md)
    
    print("Generating PDF via WeasyPrint engine...")
    try:
        # Correct md2pdf call based on help: 
        # pdf, md, etc. should be Path objects
        md2pdf(pdf=Path(output_pdf), md=temp_md_path, base_url=Path(os.getcwd()))
        print(f"Successfully generated {output_pdf}")
    except Exception as e:
        print(f"PDF Generation Failed: {e}")

if __name__ == "__main__":
    generate_pro_pdf()
