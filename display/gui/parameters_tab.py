from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class ParametersTab(ttk.Frame):
    """Separate table views for each model's parameters."""

    def __init__(self, master):
        super().__init__(master)
        
        # Meta information label
        self.lbl_meta = ttk.Label(self, text="", anchor="w")
        self.lbl_meta.pack(fill=tk.X, padx=4, pady=(4, 2))

        # Create notebook for model tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        # Initialize model tables
        self.model_tables = {}
        self._create_model_tables()

    def _create_model_tables(self):
        """Create separate tables for each model."""
        
        # SABR Parameters Table
        sabr_frame = ttk.Frame(self.notebook)
        self.notebook.add(sabr_frame, text="SABR Parameters")
        sabr_cols = ("Expiry", "alpha", "beta", "rho", "nu", "rmse", "n")
        self.model_tables["SABR"] = self._create_table(sabr_frame, sabr_cols)

        # SVI Parameters Table  
        svi_frame = ttk.Frame(self.notebook)
        self.notebook.add(svi_frame, text="SVI Parameters")
        svi_cols = ("Expiry", "a", "b", "rho", "m", "sigma", "rmse", "n")
        self.model_tables["SVI"] = self._create_table(svi_frame, svi_cols)

        # Sensitivity Analysis Table
        sens_frame = ttk.Frame(self.notebook)
        self.notebook.add(sens_frame, text="Sensitivity Analysis")
        sens_cols = ("Expiry", "atm_vol", "skew", "curv")
        self.model_tables["Sensitivity"] = self._create_table(sens_frame, sens_cols)

    def _create_table(self, parent, columns):
        """Create a treeview table with given columns."""
        tree_frame = ttk.Frame(parent)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            if col == "Expiry":
                tree.column(col, anchor=tk.W, width=100, stretch=False)
            else:
                tree.column(col, anchor=tk.E, width=80, stretch=True)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)

        return tree

    def update(self, info: Dict[str, Any] | None) -> None:
        """Update tables with latest fit information."""
        # Clear all tables
        for tree in self.model_tables.values():
            for item in tree.get_children():
                tree.delete(item)

        if not info:
            self.lbl_meta.config(text="No fit data")
            return

        # Update meta information
        parts = []
        if info.get("ticker"):
            parts.append(str(info["ticker"]))
        if info.get("asof"):
            parts.append(str(info["asof"]))
        if info.get("current_expiry"):
            parts.append(f"viewing expiry {info['current_expiry']}")
        self.lbl_meta.config(text="  ".join(parts))

        # Process data by expiry
        fit_map = info.get("fit_by_expiry") if isinstance(info, dict) else None
        if fit_map:
            self._populate_from_fit_map(fit_map)
        else:
            # Fallback: legacy single-expiry structure
            self._populate_legacy_format(info)

    def _populate_from_fit_map(self, fit_map: Dict):
        """Populate tables from fit_by_expiry structure."""
        # Collect data by model
        sabr_data = []
        svi_data = []
        sens_data = []

        for T, models in sorted(fit_map.items(), key=lambda kv: kv[0]):
            expiry = models.get("expiry") or str(T)
            
            # SABR parameters
            if "sabr" in models:
                sabr_params = models["sabr"]
                sabr_row = [expiry]
                for param in ["alpha", "beta", "rho", "nu", "rmse", "n"]:
                    val = sabr_params.get(param, "")
                    if val != "":
                        try:
                            val = f"{float(val):.6g}"
                        except (ValueError, TypeError):
                            val = str(val)
                    sabr_row.append(val)
                sabr_data.append(sabr_row)

            # SVI parameters  
            if "svi" in models:
                svi_params = models["svi"]
                svi_row = [expiry]
                for param in ["a", "b", "rho", "m", "sigma", "rmse", "n"]:
                    val = svi_params.get(param, "")
                    if val != "":
                        try:
                            val = f"{float(val):.6g}"
                        except (ValueError, TypeError):
                            val = str(val)
                    svi_row.append(val)
                svi_data.append(svi_row)

            # Sensitivity parameters
            if "sens" in models:
                sens_params = models["sens"]
                sens_row = [expiry]
                for param in ["atm_vol", "skew", "curv"]:
                    val = sens_params.get(param, "")
                    if val != "":
                        try:
                            val = f"{float(val):.6g}"
                        except (ValueError, TypeError):
                            val = str(val)
                    sens_row.append(val)
                sens_data.append(sens_row)

        # Populate tables
        for row in sabr_data:
            self.model_tables["SABR"].insert("", tk.END, values=row)
        for row in svi_data:
            self.model_tables["SVI"].insert("", tk.END, values=row)
        for row in sens_data:
            self.model_tables["Sensitivity"].insert("", tk.END, values=row)

    def _populate_legacy_format(self, info: Dict):
        """Populate tables from legacy single-expiry format."""
        expiry = info.get("expiry", "")

        # SABR
        if "sabr" in info:
            sabr_params = info["sabr"]
            sabr_row = [expiry]
            for param in ["alpha", "beta", "rho", "nu", "rmse", "n"]:
                val = sabr_params.get(param, "")
                if val != "":
                    try:
                        val = f"{float(val):.6g}"
                    except (ValueError, TypeError):
                        val = str(val)
                sabr_row.append(val)
            self.model_tables["SABR"].insert("", tk.END, values=sabr_row)

        # SVI
        if "svi" in info:
            svi_params = info["svi"]
            svi_row = [expiry]
            for param in ["a", "b", "rho", "m", "sigma", "rmse", "n"]:
                val = svi_params.get(param, "")
                if val != "":
                    try:
                        val = f"{float(val):.6g}"
                    except (ValueError, TypeError):
                        val = str(val)
                svi_row.append(val)
            self.model_tables["SVI"].insert("", tk.END, values=svi_row)

        # Sensitivity
        if "sens" in info:
            sens_params = info["sens"]
            sens_row = [expiry]
            for param in ["atm_vol", "skew", "curv"]:
                val = sens_params.get(param, "")
                if val != "":
                    try:
                        val = f"{float(val):.6g}"
                    except (ValueError, TypeError):
                        val = str(val)
                sens_row.append(val)
            self.model_tables["Sensitivity"].insert("", tk.END, values=sens_row)


# Example usage and testing
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Model Parameters - Separated View")
    root.geometry("800x600")

    # Sample data structure
    sample_data = {
        "ticker": "SPX",
        "asof": "2025-08-21",
        "current_expiry": "2025-09-12",
        "fit_by_expiry": {
            "2025-09-05": {
                "expiry": "2025-09-05",
                "sabr": {"n": 80},
                "sens": {"atm_vol": 0.286192, "skew": -0.0715535, "curv": 0.0119297}
            },
            "2025-09-12": {
                "expiry": "2025-09-12", 
                "sabr": {"alpha": 5, "beta": 0.5, "rho": -0.999, "nu": 1e-06, "rmse": 8.8257e-08, "n": 65},
                "svi": {"a": 1e-12, "b": 0.102131, "rho": 0.0155951, "m": -0.00339881, "sigma": 0.061342, "rmse": 0.104224, "n": 65},
                "sens": {"atm_vol": 0.286197, "skew": -0.071557, "curv": 0.0119322}
            },
            "2025-09-19": {
                "expiry": "2025-09-19",
                "sabr": {"alpha": 5, "beta": 0.5, "rho": -0.999, "nu": 1e-06, "rmse": 4.87367e-09, "n": 113},
                "svi": {"a": 0.00302272, "b": 0.109041, "rho": -0.0111255, "m": -0.063805, "sigma": 1e-08, "rmse": 0.228105, "n": 113},
                "sens": {"atm_vol": 0.286201, "skew": -0.0715805, "curv": 0.0119347}
            }
        }
    }

    params_tab = ParametersTab(root)
    params_tab.pack(fill=tk.BOTH, expand=True)
    params_tab.update(sample_data)

    root.mainloop()