# Ticket 03: Dismantle Global Run Card

**Blockers:** None
**Status:** To Do

**Description:**
The global "Run Card" on the left panel creates a confusing button hierarchy by displaying all algorithm run buttons simultaneously. 

**Tasks:**
- In `codes/gui/main_window/stochastic_mixin.py`, locate the `run-card` QFrame creation logic.
- Completely remove the `run-card` container and its layout from the left panel.
- Ensure that removing this does not break the layout of the `content_splitter`.
