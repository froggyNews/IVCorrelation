import math


def norm_cdf(x):
    """Cumulative distribution function for the standard normal distribution."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x):
    """Probability density function for the standard normal distribution."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def black_scholes_call(S, K, T, r, sigma):
    """Price a European call option using the Black-Scholes model."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        # deterministic case
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)


def bachelier_call(S, K, T, r, sigma):
    """Price a European call option using the Bachelier (normal) model."""
    if T <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        return max(S - K * math.exp(-r * T), 0.0)
    forward = S * math.exp(r * T)
    d = (forward - K) / (sigma * math.sqrt(T))
    price = (forward - K) * norm_cdf(d) + sigma * math.sqrt(T) * norm_pdf(d)
    return math.exp(-r * T) * price


# Volatility surface examples

def f_low(K_star, T_star):
    """Conservative volatility estimate as a function of strike and maturity."""
    base_vol = 0.2
    adjustment = -0.05 * (T_star / (T_star + 1.0))
    return max(0.0001, base_vol + adjustment)


def f_med(K_star, T_star):
    """Baseline volatility estimate."""
    base_vol = 0.2
    return base_vol


def f_high(K_star, T_star):
    """Aggressive volatility estimate."""
    base_vol = 0.2
    adjustment = 0.05 * (T_star / (T_star + 1.0))
    return base_vol + adjustment


def _price_with_model(S, K, T, r, sigma):
    """Choose pricing model based on inputs."""
    if abs(r) < 0.01:
        return bachelier_call(S, K, T, r, sigma)
    return black_scholes_call(S, K, T, r, sigma)


def price_options(K_star, T_star, S, r):
    """Return conservative, baseline, and aggressive option prices."""
    sigma_low = f_low(K_star, T_star)
    sigma_med = f_med(K_star, T_star)
    sigma_high = f_high(K_star, T_star)

    price_low = _price_with_model(S, K_star, T_star, r, sigma_low)
    price_med = _price_with_model(S, K_star, T_star, r, sigma_med)
    price_high = _price_with_model(S, K_star, T_star, r, sigma_high)

    return {
        "conservative": price_low,
        "baseline": price_med,
        "aggressive": price_high,
    }


if __name__ == "__main__":
    # Example usage
    K = 100
    T = 1.0  # 1 year
    S0 = 105
    r = 0.05
    prices = price_options(K, T, S0, r)
    for k, v in prices.items():
        print(f"{k}: {v:.4f}")
