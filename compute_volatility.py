import argparse
import pandas as pd
import numpy as np
from pysabr import Hagan2002LognormalSABR, Hagan2002NormalSABR
from sabr import hagan_lognormal_vol
from svi import fit_svi_smile, svi_implied_vol
from fit_spline import fit_surfaces


def fit_sabr_smile(strikes, vols, f, t, beta=1.0):
    """Fit SABR parameters to a volatility smile with improved fitting.
    
    Parameters
    ----------
    strikes : array-like
        Strike prices
    vols : array-like
        Implied volatilities
    f : float
        Forward price
    t : float
        Time to maturity
    beta : float, optional
        SABR beta parameter (default: 1.0 for log-normal)
    
    Returns
    -------
    tuple
        (alpha, beta, rho, nu) SABR parameters
    """
    strikes = np.array(strikes)
    vols = np.array(vols)
    
    # Filter out invalid data
    valid_mask = (strikes > 0) & (vols > 0) & ~np.isnan(vols)
    if valid_mask.sum() < 3:  # Lowered from 5 to 3 for smaller stocks
        print(f"Warning: Insufficient valid data points ({valid_mask.sum()}) for SABR fitting")
        return None
    
    strikes = strikes[valid_mask]
    vols = vols[valid_mask]
    
    # Sort by strike for better fitting
    sort_idx = np.argsort(strikes)
    strikes = strikes[sort_idx]
    vols = vols[sort_idx]
    
    # Better initial parameter estimation
    # Alpha: Use ATM volatility as initial guess
    atm_idx = np.argmin(np.abs(strikes - f))
    alpha_init = vols[atm_idx] * np.sqrt(t)
    
    # Rho: Estimate from skew
    if len(strikes) >= 3:
        # Use linear regression to estimate skew
        log_moneyness = np.log(strikes / f)
        skew = np.polyfit(log_moneyness, vols, 1)[0]
        rho_init = np.clip(-skew * 0.5, -0.9, 0.9)  # Clip to reasonable range
    else:
        rho_init = 0.0
    
    # Nu: Estimate from volatility of volatility
    if len(vols) >= 3:
        vol_std = np.std(vols)
        nu_init = np.clip(vol_std * 2.0, 0.1, 3.0)  # Scale and clip
    else:
        nu_init = 0.5
    
    # Parameter bounds (more reasonable)
    bounds = [
        (0.01, 2.0),     # alpha: positive, reasonable range
        (-0.9, 0.9),     # rho: correlation between -0.9 and 0.9
        (0.05, 2.0)      # nu: volatility of volatility
    ]
    
    # Multiple optimization attempts with different starting points
    best_result = None
    best_error = float('inf')
    
    # Starting points for optimization
    starting_points = [
        [alpha_init, rho_init, nu_init],
        [alpha_init * 0.8, rho_init, nu_init],
        [alpha_init * 1.2, rho_init, nu_init],
        [alpha_init, rho_init * 0.8, nu_init],
        [alpha_init, rho_init, nu_init * 0.8],
        [alpha_init, rho_init, nu_init * 1.2],
    ]
    
    def objective(params):
        alpha, rho, nu = params
        try:
            fitted_vols = [hagan_lognormal_vol(f, k, t, alpha, beta, rho, nu) for k in strikes]
            # Use relative error for better fit
            relative_errors = np.abs((vols - fitted_vols) / vols)
            return np.mean(relative_errors)
        except:
            return 1e10  # Return large error if fitting fails
    
    try:
        from scipy.optimize import minimize
        
        for i, x0 in enumerate(starting_points):
            try:
                result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B', 
                               options={'maxiter': 1000})
                
                if result.success:
                    error = result.fun
                    if error < best_error:
                        best_error = error
                        best_result = result
                        
            except Exception as e:
                continue
        
        if best_result is not None and best_error < 0.5:  # Accept if relative error < 50%
            alpha, rho, nu = best_result.x
            
            # Validate parameters
            if (0.01 <= alpha <= 2.0 and -0.9 <= rho <= 0.9 and 0.05 <= nu <= 2.0):
                return alpha, beta, rho, nu
            else:
                print(f"Warning: SABR parameters out of bounds: alpha={alpha:.4f}, rho={rho:.4f}, nu={nu:.4f}")
                return None
        else:
            print(f"Warning: SABR optimization failed or poor fit (error: {best_error:.4f})")
            return None
            
    except Exception as e:
        print(f"Warning: SABR fitting failed: {e}")
        return None


def weighted_stats(df, *, fit_method: str = "sabr"):
    """Compute weighted stats for each T (maturity).

    The DataFrame must contain columns:
    - K
    - T
    - sigma
    - corr_weight
    - liq_weight
    """
    # filter out rows with missing sigma
    df = df.dropna(subset=['sigma'])
    
    # Additional filtering for data quality
    df = df[df['sigma'] > 0.01]  # Remove very low IV
    df = df[df['sigma'] < 2.0]   # Remove very high IV
    df = df[df['T'] > 0.01]      # Remove very short maturities

    # total weight is correlation weight times liquidity weight
    df['weight'] = df['corr_weight'] * df['liq_weight']

    def agg(group):
        w = group['weight']
        s = group['sigma']
        w_sum = w.sum()
        mean = (w * s).sum() / w_sum if w_sum > 0 else float('nan')
        variance = (w * (s - mean) ** 2).sum() / w_sum if w_sum > 0 else float('nan')
        std = variance ** 0.5 if w_sum > 0 else float('nan')
        
        strikes = group['K'].values
        vols = group['sigma'].values
        f = strikes.mean() if len(strikes) > 0 else float('nan')
        t = group['T'].iloc[0] if len(group) > 0 else float('nan')

        if fit_method == "svi":
            params = fit_svi_smile(strikes, vols, f, t)
            if params is None:
                a = b = rho = m = sigma = float('nan')
            else:
                a, b, rho, m, sigma = params
            return pd.Series({
                'sigma_med': mean,
                'sigma_std': std,
                'sigma_low': s.min() if len(s) > 0 else float('nan'),
                'sigma_high': s.max() if len(s) > 0 else float('nan'),
                'svi_a': a,
                'svi_b': b,
                'svi_rho': rho,
                'svi_m': m,
                'svi_sigma': sigma,
            })
        else:
            sabr_params = fit_sabr_smile(strikes, vols, f, t)
            if sabr_params is not None:
                alpha, beta, rho, nu = sabr_params
            else:
                alpha = beta = rho = nu = float('nan')
            return pd.Series({
                'sigma_med': mean,
                'sigma_std': std,
                'sigma_low': s.min() if len(s) > 0 else float('nan'),
                'sigma_high': s.max() if len(s) > 0 else float('nan'),
                'sabr_alpha': alpha,
                'sabr_beta': beta,
                'sabr_rho': rho,
                'sabr_nu': nu,
            })

    # Group by T (maturity) only
    result = df.groupby('T').apply(agg).reset_index()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Compute weighted volatility statistics with surface fitting"
    )
    parser.add_argument(
        "csv",
        help="CSV file with columns K,T,sigma,corr_weight,liq_weight",
    )
    parser.add_argument("-o", "--output", help="Output CSV or pickle file")
    parser.add_argument(
        "--fit-method",
        choices=["sabr", "svi", "tps"],
        default="sabr",
        help="Fitting method: SABR (default), SVI, or thin-plate spline",
    )
    parser.add_argument(
        "--sabr-only",
        action="store_true",
        help="Output only SABR parameters (if using SABR)",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.csv)

    if args.fit_method == "tps":
        pts = [tuple(row) for row in data[["K", "T", "sigma"]].values]
        surfaces = fit_surfaces(pts, pts, pts)
        if args.output:
            import pickle

            with open(args.output, "wb") as f:
                pickle.dump(surfaces, f)
        else:
            k0 = float(data["K"].median())
            t0 = float(data["T"].median())
            print(
                f"Median surface value at K={k0:.2f}, T={t0:.2f}: {surfaces[1](k0, t0):.4f}"
            )
        return

    result = weighted_stats(data, fit_method=args.fit_method)

    if args.fit_method == "sabr" and args.sabr_only:
        sabr_cols = ["T", "sabr_alpha", "sabr_beta", "sabr_rho", "sabr_nu"]
        result = result[sabr_cols].dropna()

    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_csv(index=False))


if __name__ == '__main__':
    main()
