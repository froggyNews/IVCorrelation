# def _local_poly_fit_atm(k: np.ndarray, iv: np.ndarray, weights: Optional[np.ndarray] = None,
#                         band: float = 0.25) -> Dict[str, float]:
#     """Quadratic in k around ATM (k=0). Returns f(0), f'(0), f''(0) and rmse."""
#     # focus near-ATM
#     mask = np.abs(k) <= band
#     if mask.sum() < 3:
#         # widen if too sparse
#         mask = np.argsort(np.abs(k))[:max(3, min(7, k.size))]
#     x = k[mask]; y = iv[mask]
#     if weights is not None:
#         w = weights[mask]
#         W = np.diag(w)
#         X = np.column_stack([np.ones_like(x), x, x**2])
#         beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
#     else:
#         X = np.column_stack([np.ones_like(x), x, x**2])
#         beta = np.linalg.lstsq(X, y, rcond=None)[0]
#     # f(k) = a + b k + c k^2
#     a, b, c = beta
#     yhat = a + b*x + c*x**2
#     rmse = float(np.sqrt(np.mean((yhat - y)**2)))
#     return {"atm_vol": float(a), "skew": float(b), "curv": float(2*c), "rmse": rmse}

# def _predict_with_svi(params: Tuple[float, float, float, float, float], k: np.ndarray, T: float) -> np.ndarray:
#     # raw-SVI total variance: w(k) = a + b( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )
#     a, b, rho, m, sigma = params
#     w = a + b * (rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))
#     w = np.clip(w, 1e-10, None)
#     return np.sqrt(w / max(T, 1e-12))

# def _finite_diff(f, x0: float, h: float = 1e-3) -> Tuple[float, float]:
#     # return f'(x0) and f''(x0)
#     f_p = f(x0 + h); f_m = f(x0 - h); f0 = f(x0)
#     first = (f_p - f_m) / (2*h)
#     second = (f_p - 2*f0 + f_m) / (h*h)
#     return float(first), float(second)