"""Command-line entry point for the volatility surface pipeline."""

from importlib import import_module

STAGES = [
    "data_collection.options:fetch_option_chain",
    "preprocessing.vol_surface:build_vol_surface",
    "aggregation.theme_surface:aggregate_surfaces",
    "smoothing.pricing:smooth_surface",
    "validation.backtest:run_backtest",
]

def main():
    print("Starting volatility surface pipeline...")
    for stage in STAGES:
        module_path, func_name = stage.split(":")
        module = import_module(f"volsurf.{module_path}")
        func = getattr(module, func_name)
        try:
            print(f"Running {func.__name__}()")
            func(None)  # placeholder call
        except NotImplementedError as exc:
            print(f"{func.__name__} not implemented: {exc}")
    print("Pipeline complete.")

if __name__ == "__main__":
    main()
