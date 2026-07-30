# Ticket 04: Implement Contextual Run Buttons

**Blockers:** 03-dismantle-global-run-card
**Status:** To Do

**Description:**
Since the global run card is gone, users need a way to run the algorithms. These buttons must become contextual.

**Tasks:**
- Move each "Run" button (e.g., `self.run_ga_button`, `self.run_pso_button`) into their respective optimization mixin tabs (e.g., inside the GA tab in `ga_mixin.py`, the PSO tab in `pso_mixin.py`, etc.).
- Ensure that each tab has exactly ONE `#primary-button` (the run button).
- Ensure that all other buttons in these tabs (like "Save Results", "Reset Parameters") use the `#secondary-button` style (ghost/outline).
