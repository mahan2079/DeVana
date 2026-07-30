## Problem Statement
The DeVana software currently suffers from a cluttered, "gamer dashboard" aesthetic. The widespread use of glowing neon gradients, inconsistent 8px-12px rounded corners, heavy group box borders, and drop shadows creates severe visual noise and cognitive overload. Furthermore, the global "Run Card" clusters all execution actions into a single sidebar with identical primary button styles, forcing users to read through a wall of identical options rather than providing contextual paths.

## Solution
Redesign the GUI (built with PyQt5) following a "Refined Industrial" aesthetic. We will completely strip out all gradient backgrounds and borders, replacing them with solid, perceptually uniform dark grays. The structural containment will shift from thick group box lines to spatial whitespace, enforced by a strict, modern 6px border radius. Functionally, the global run card will be dismantled, and execution buttons will be moved directly into their relevant optimization tabs, utilizing strict primary/secondary styling to reduce cognitive load.

## User Stories
1. As an engineer, I want the interface to use solid, muted backgrounds so that my focus is drawn to the data and plots, not the UI chrome.
2. As a researcher, I want exactly one primary accent color for active states so that I instantly know where I am and what is selected.
3. As a user configuring parameters, I want sections separated by whitespace and subtle background shifts rather than heavy borders, so that the layout breathes and is easier to scan.
4. As a user, I want a consistent, sharp 6px corner radius across all inputs and panels so that the tool feels precise and professionally engineered.
5. As an engineer running an optimization, I want the "Run" button to be located inside the specific algorithm's tab, so that the action is contextual to the parameters I am currently tweaking.
6. As a user, I want secondary actions (like "Save Plot" or "Reset") to be styled as ghost/outline buttons, so that they do not compete visually with the primary "Run" action.

## Implementation Decisions
- **theme_mixin.py:** Will be completely overhauled. We will remove all `qlineargradient` syntax and hardcoded dropshadows. The dark theme base will be `#1E1E28`, and the primary accent color will be restricted to Teal (`#009688`).
- **stochastic_mixin.py:** The global `#run-card` QFrame will be deleted from the left panel sidebar. Typography sizes (like the 28pt banner title) will be rebalanced to 20pt.
- **Algorithm Mixins (ga_mixin, pso_mixin, etc.):** The specific "Run" buttons will be relocated to the bottom of these tabs.
- **Button Styling:** We will strictly enforce `#primary-button` QSS rules for the contextual run buttons, and `#secondary-button` rules for auxiliary actions.

## Testing Decisions
- **Primary Seam:** The boundary between the `MainWindow` UI elements and the QThread `workers/` (e.g., `run_ga`, `run_pso`). We are purely moving the buttons; we must ensure the `clicked.connect()` signal routing remains intact.
- **Visual QA:** Since this is a PyQt5 stylistic rewrite, testing will primarily rely on manual visual inspection by running `codes/run.py` to ensure the layout hasn't broken, the tabs still scale correctly in the `content_splitter`, and the CSS/QSS rules render identically across different operating systems.
- **Good Tests:** A successful test will show that a user can configure GA parameters in the GA tab, click the newly relocated "Run GA" button, and the worker executes properly without any global state errors.

## Out of Scope
- Modifying the underlying mathematical models, FRF logic, or algorithm implementations.
- Refactoring the core worker threads (e.g., `GAWorker`, `PSOWorker`).
- Changing the backend plotting logic (Matplotlib integration), aside from ensuring they sit cleanly in the new spatial layout.

## Further Notes
- ADR-0001 (`docs/adr/0001-ui-ux-redesign-principles.md`) serves as the permanent record for these aesthetic decisions.

**Triage Label:** `ready-for-agent`
