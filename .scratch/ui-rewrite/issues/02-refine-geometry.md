# 02 — Refine Geometry & Typography

**What to build:** A modern, spatial layout. The user will navigate an interface separated by whitespace and subtle background shifts rather than thick borders, and all elements will have precise, matching corner radii.

**Blocked by:** 01 — Purge Gradients & Establish Baseline Colors

**Status:** ready-for-agent

- [ ] Thick borders are removed from `QGroupBox` elements in `theme_mixin.py`.
- [ ] A global, strict `border-radius: 6px` is enforced on all interactive elements (tabs, buttons, inputs).
- [ ] The massive 28pt banner title in `stochastic_mixin.py` is reduced to a balanced size (e.g., 20pt).
- [ ] Running the application confirms the layout relies on spatial containment and looks precise.
