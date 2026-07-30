# 04 — Dismantle Global Run Card (Contract)

**What to build:** The final, clean UI. The user will no longer see a confusing, overwhelming global list of identical "Run" buttons on the left sidebar.

**Blocked by:** 03 — Implement Contextual Execution Buttons (Expand)

**Status:** ready-for-agent

- [ ] The global `run-card` QFrame and its entire layout logic are deleted from `stochastic_mixin.py`.
- [ ] The layout of the `content_splitter` successfully scales and looks natural without the left-hand run card.
- [ ] The application launches successfully and runs optimizations perfectly relying solely on the new contextual buttons.
