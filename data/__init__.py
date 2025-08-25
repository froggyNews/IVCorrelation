"""
Data module for CleanIV_Correlation project.

This module contains data handling, downloading, and database utilities.
"""

# Import key functions for easier access
from .ticker_groups import (
    save_ticker_group, load_ticker_group, list_ticker_groups,
    delete_ticker_group, create_default_groups
)
from .db_utils import get_conn, ensure_initialized, ensure_indexes, insert_features
from .data_downloader import save_for_tickers
from .underlying_prices import update_underlying_prices
from .interest_rates import (
    save_interest_rate, load_interest_rate, get_default_interest_rate,
    list_interest_rates, delete_interest_rate, set_default_interest_rate,
    get_interest_rate_names, create_default_interest_rates
)
from .feature_engineering import (
    build_pooled_iv_return_dataset_time_safe,
    build_iv_return_dataset_time_safe,
    build_target_peer_dataset,
    add_all_features,
    build_iv_panel,
    finalize_dataset,
)
# Replace broken absolute import with safe relative import
try:
    from .data_loader_coordinator import (
        DataCoordinator,
        load_cores_with_auto_fetch,
        validate_cores,
    )
except Exception:  # pragma: no cover
    def _missing(*args, **kwargs):  # type: ignore
        raise RuntimeError("data_loader_coordinator not available")
    DataCoordinator = _missing  # type: ignore
    load_cores_with_auto_fetch = _missing  # type: ignore
    validate_cores = _missing  # type: ignore


__all__ = [
    'save_ticker_group', 'load_ticker_group', 'list_ticker_groups',
    'delete_ticker_group', 'create_default_groups',
    'get_conn', 'ensure_initialized', 'ensure_indexes', 'save_for_tickers',
    'insert_features',
    'update_underlying_prices',
    'save_interest_rate', 'load_interest_rate', 'get_default_interest_rate',
    'list_interest_rates', 'delete_interest_rate', 'set_default_interest_rate',
    'get_interest_rate_names', 'create_default_interest_rates',
    'build_pooled_iv_return_dataset_time_safe',
    'build_iv_return_dataset_time_safe',
    'build_target_peer_dataset',
    'add_all_features',
    'build_iv_panel',
    'finalize_dataset',
    'DataCoordinator',
    'load_cores_with_auto_fetch',
    'validate_cores',
]
