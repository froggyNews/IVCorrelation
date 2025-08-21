from typing import Optional, List
import numpy as np
import pandas as pd
from .equal import equal_weights


def open_interest_weights(peers_list: List[str], asof: Optional[str]) -> pd.Series:
    """Compute weights proportional to open interest on ``asof`` date."""
    from data.db_utils import get_conn

    if not peers_list:
        return pd.Series(dtype=float)
    conn = get_conn()
    if asof is None:
        row = conn.execute(
            "SELECT MAX(asof_date) FROM options_quotes WHERE ticker IN ({})".format(
                ",".join("?" * len(peers_list))
            ),
            peers_list,
        ).fetchone()
        asof = row[0] if row and row[0] else None
        if asof is None:
            return equal_weights(peers_list)
    q = (
        "SELECT ticker, SUM(open_interest) AS oi FROM options_quotes "
        "WHERE asof_date = ? AND ticker IN ({}) GROUP BY ticker"
    ).format(",".join("?" * len(peers_list)))
    df = pd.read_sql_query(q, conn, params=[asof] + peers_list)
    if df.empty:
        return equal_weights(peers_list)
    s = pd.Series(df["oi"].values, index=df["ticker"].str.upper())
    total = float(s.sum())
    if not np.isfinite(total) or total <= 0:
        return equal_weights(peers_list)
    return (s / total).reindex([p.upper() for p in peers_list]).fillna(0.0)
