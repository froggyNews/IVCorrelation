from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class ParametersTab(ttk.Frame):
    """Simple table view for model and sensitivity parameters."""

    def __init__(self, master):
        super().__init__(master)
        cols = ("Model", "Parameter", "Value")
        self.tree = ttk.Treeview(self, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.W, stretch=True)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def update(self, info: Dict[str, Any] | None) -> None:
        """Update table with latest fit information."""
        for i in self.tree.get_children():
            self.tree.delete(i)
        if not info:
            return

        def insert(model: str, params: Dict[str, Any]):
            for k, v in params.items():
                try:
                    val = float(v)
                except Exception:
                    continue
                self.tree.insert("", tk.END, values=(model, k, f"{val:.6g}"))

        if info.get("svi"):
            insert("SVI", info["svi"])
        if info.get("sabr"):
            insert("SABR", info["sabr"])
        if info.get("sens"):
            insert("Sensitivity", info["sens"])
