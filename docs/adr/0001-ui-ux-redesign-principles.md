# ADR 0001: UI/UX Redesign Principles (Refined Industrial)

## Status
Accepted

## Context
DeVana is a heavy-duty scientific optimization framework used by engineers and researchers. The initial UI implementation (PyQt5) utilized heavy "AI-generated" aesthetics—widespread use of `qlineargradient` for backgrounds and borders, neon accent colors (cyan, teal, pink), excessive drop shadows, and heavy physical containment (borders and group boxes). This resulted in visual clutter, cognitive overload, and an unprofessional "gamer dashboard" appearance that detracted from the software's scientific purpose.

We needed a unified design system that communicates precision, allows dense data to breathe, and strictly guides the user's focus.

## Decision
We are completely rewriting the UI/UX according to a **Refined/Industrial Minimalist** design system with the following core pillars:

1. **Aesthetic Direction (No Gradients):** 
   - All `qlineargradient` definitions are stripped from the PyQt5 stylesheets.
   - Backgrounds will use solid, subdued, perceptually uniform dark grays (e.g., `#1E1E28`).
   - Exactly ONE vibrant accent color (e.g., teal `#009688`) will be reserved strictly for the active tab, the single primary action button, and active boolean states.

2. **Containment and Geometry (Spatial Layout):**
   - Shift from "border-based containment" to "spatial containment." Remove thick `QGroupBox` borders.
   - Separate functional sections (Inputs vs. Results) using distinct background tints and generous whitespace.
   - Enforce a strict, sharp `6px` border radius globally, replacing the soft, varying 8px-12px radii.
   - Use bold typography to denote sections instead of wrapping everything in a bordered box.

3. **Action Hierarchy (Contextual Execution):**
   - Remove the global "Run Card" from the left sidebar.
   - Execution actions (e.g., "Run GA", "Run PSO") are now **contextual**, appearing only inside their respective optimization tabs.
   - Implement strict button hierarchy: exactly one solid-filled Primary button per view. All secondary actions ("Save Plot", "Reset") must be ghost or outline buttons.

## Consequences
- **Positive:** The UI will instantly look like a professional, high-end engineering tool. Cognitive load is reduced because the user no longer has to visually filter through 8 identical glowing primary buttons. The code in `theme_mixin.py` and `stochastic_mixin.py` will be significantly cleaner and easier to maintain.
- **Negative:** Moving away from a global run menu means users must navigate to the specific algorithm tab before clicking run. This requires rewiring some of the signal/slot connections that previously assumed global access to run actions.
