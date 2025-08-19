from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class ParametersTab(ttk.Frame):
    """Simple table view for model and sensitivity parameters."""

    def __init__(self, master):
        super().__init__(master)
        self.lbl_meta = ttk.Label(self, text="", anchor="w")
        self.lbl_meta.pack(fill=tk.X, padx=4, pady=(4, 2))

        cols = ("Model", "Parameter", "Value")
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(tree_frame, columns=cols, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor=tk.W, stretch=True)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

    def update(self, info: Dict[str, Any] | None) -> None:
        """Update table with latest fit information."""
        for i in self.tree.get_children():
            self.tree.delete(i)
        if not info:
            self.lbl_meta.config(text="No fit data")
            return

        parts = []
        if info.get("ticker"):
            parts.append(str(info["ticker"]))
        if info.get("asof"):
            parts.append(str(info["asof"]))
        if info.get("expiry"):
            parts.append(f"expiry {info['expiry']}")
        self.lbl_meta.config(text="  ".join(parts))

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
