from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class InputManager:
    """Lightweight store for GUI settings.

    The GUI updates this manager whenever a control changes so that callers
    can grab a coherent snapshot of the current configuration without polling
    each widget individually. This reduces the delay between editing a field
    and using the new values for plotting or data ingestion."""

    settings: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> None:
        """Merge provided key/value pairs into the settings store."""
        self.settings.update(kwargs)

    def as_dict(self) -> Dict[str, Any]:
        """Return a shallow copy of all current settings."""
        return dict(self.settings)
