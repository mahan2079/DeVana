# 01 — Purge Gradients & Establish Baseline Colors

**What to build:** A flattened, professional UI base. The user will see solid dark backgrounds and a single strictly enforced teal accent color, rather than a muddy UI filled with glowing gradients.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] All `qlineargradient` rules are removed from `theme_mixin.py` (both light and dark themes).
- [ ] The dark theme relies on solid, subdued dark grays (e.g., `#1E1E28`) for backgrounds.
- [ ] A single primary accent color (`#009688`) is used strictly for active states and primary buttons.
- [ ] Running the application visually confirms the absence of neon gradients.
