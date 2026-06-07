# SidebarMixin

## Purpose
Core implementation of SidebarMixin logic.

## Internal Logic Flow: `create_sidebar`
```mermaid
graph TD
    Start["Start: create_sidebar"] --> Step0["Initialize sidebar_container"]
    Step0["Initialize sidebar_container"] --> Step1["Initialize sidebar_layout"]
    Step1["Initialize sidebar_layout"] --> Step2["Initialize logo_container"]
    Step2["Initialize logo_container"] --> Step3["Initialize logo_layout"]
    Step3["Initialize logo_layout"] --> Step4["Initialize title"]
    Step4["Initialize title"] --> Step5["Initialize version"]
    Step5["Initialize version"] --> Step6["Initialize line"]
    Step6["Initialize line"] --> Step7["Initialize nav_container"]
    Step7["Initialize nav_container"] --> End["End: create_sidebar"]
```

### Flowchart Pseudo-code
```python
FUNCTION create_sidebar(self, BEAM_IMPORTS_SUCCESSFUL):
    DO "Initialize sidebar_container"
    DO "Initialize sidebar_layout"
    DO "Initialize logo_container"
    DO "Initialize logo_layout"
    DO "Initialize title"
    DO "Initialize version"
    DO "Initialize line"
    DO "Initialize nav_container"
END FUNCTION
```

## Methods & Functions

### `create_sidebar`
- **Arguments**: `self, BEAM_IMPORTS_SUCCESSFUL`
- **Returns**: `None`
- **Logic**: Assigns sidebar_container; Assigns sidebar_layout; Assigns logo_container; Assigns logo_layout; Assigns title...

### `change_page`
- **Arguments**: `self, index`
- **Returns**: `None`
- **Logic**: Loops over [self.intro_btn, self.stochast; Conditional: index == 0

### `toggle_theme`
- **Arguments**: `self`
- **Returns**: `None`
- **Logic**: Conditional: self.current_theme == 'Dark'

