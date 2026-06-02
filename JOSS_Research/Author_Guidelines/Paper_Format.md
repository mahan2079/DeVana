# JOSS Paper Format Guidelines

JOSS papers are short (250–1000 words) and focus on the software's research application.

## 1. The `paper.md` File
The paper is written in Markdown with a YAML front matter.

### Mandatory Sections:
- **Summary:** A high-level description for a non-specialist audience.
- **Statement of Need:** Explains why the software was created and what problem it solves. Compare it to existing tools.
- **State of the Field:** A comparison to other related software packages.
- **Research Impact Statement:** Evidence of realized or potential impact.
- **AI Usage Disclosure:** Transparent disclosure of any generative AI used in development or writing.
- **References:** Citations using BibTeX (linked via `paper.bib`).

## 2. Citations and References
- All citations in `paper.md` should be in the format `[@citation_key]`.
- All archival references must include **DOIs** whenever possible.
- Use a separate `paper.bib` file for BibTeX entries.

## 3. Formatting Rules
- Use standard Markdown (CommonMark).
- LaTeX math is supported (e.g., `$x^2$`).
- Figures can be included using standard Markdown syntax.

## 4. Word Count
- Aim for **250 to 1,000 words**.
- Keep the focus on the **software**, not the new research results obtained using it (those belong in other journals).

---
*Source: JOSS Documentation (2026)*
