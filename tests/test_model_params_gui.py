import os
import tkinter as tk
import pytest

from display.gui.model_params_gui import ModelParamsFrame


@pytest.mark.skipif(os.environ.get("DISPLAY", "") == "", reason="requires X11 display")
def test_param_panel_multi_ticker():
    root = tk.Tk()
    root.withdraw()
    frame = ModelParamsFrame(root)
    frame.ent_ticker.insert(0, "QQQ,UNH")
    frame.cmb_model.set("sabr")
    frame.ent_param.insert(0, "alpha")
    frame._plot()
    rows = [frame.table.tree.item(i)["values"] for i in frame.table.tree.get_children()]
    assert len(rows) == 2
    params = [r[1] for r in rows]
    assert any("QQQ" in p for p in params)
    assert any("UNH" in p for p in params)
    root.destroy()
