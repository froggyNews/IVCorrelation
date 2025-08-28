def is_smile_active(self) -> bool:
    return (
        self._current_plot_type is not None
        and self._current_plot_type.startswith("Smile")
        and self._smile_ctx is not None
    )

def next_expiry(self):
    if not self.is_smile_active():
        return
    Ts = self._smile_ctx["Ts"]
    self._smile_ctx["idx"] = min(self._smile_ctx["idx"] + 1, len(Ts) - 1)
    self._render_smile_at_index()

def prev_expiry(self):
    if not self.is_smile_active():
        return
    Ts = self._smile_ctx["Ts"]
    self._smile_ctx["idx"] = max(self._smile_ctx["idx"] - 1, 0)
    self._render_smile_at_index()
