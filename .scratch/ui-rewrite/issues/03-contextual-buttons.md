# 03 — Implement Contextual Execution Buttons (Expand)

**What to build:** Contextual actions for running algorithms. The user will see a clearly defined "Run" button directly inside the optimization tab they are configuring, making the action immediately obvious.

**Blocked by:** None — can start immediately.

**Status:** ready-for-agent

- [ ] A new "Run" button is added inside the specific optimization tabs (e.g., GA tab, PSO tab, etc.).
- [ ] The newly placed buttons correctly trigger their respective backend worker methods when clicked (e.g., `self.run_ga`).
- [ ] The contextual run button uses the `#primary-button` QSS style, while all other local actions use the `#secondary-button` style.
- [ ] Note: The old global buttons in the left sidebar will temporarily remain active and visible.
