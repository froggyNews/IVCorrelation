import argparse
import pandas as pd


def weighted_stats(df):
    """Compute weighted stats for each (K, T).

    The DataFrame must contain columns:
    - K
    - T
    - sigma
    - corr_weight
    - liq_weight
    """
    # filter out rows with missing sigma
    df = df.dropna(subset=['sigma'])

    # total weight is correlation weight times liquidity weight
    df['weight'] = df['corr_weight'] * df['liq_weight']

    def agg(group):
        w = group['weight']
        s = group['sigma']
        w_sum = w.sum()
        if w_sum == 0:
            return pd.Series({
                'sigma_med': float('nan'),
                'sigma_std': float('nan'),
                'sigma_low': float('nan'),
                'sigma_high': float('nan'),
            })
        mean = (w * s).sum() / w_sum
        variance = (w * (s - mean) ** 2).sum() / w_sum
        std = variance ** 0.5
        return pd.Series({
            'sigma_med': mean,
            'sigma_std': std,
            'sigma_low': mean - std,
            'sigma_high': mean + std,
        })

    return df.groupby(['K', 'T']).apply(agg).reset_index()


def main():
    parser = argparse.ArgumentParser(description="Compute weighted volatility statistics")
    parser.add_argument('csv', help='CSV file with columns K,T,sigma,corr_weight,liq_weight')
    parser.add_argument('-o', '--output', help='Output CSV file (optional)')
    args = parser.parse_args()

    data = pd.read_csv(args.csv)
    result = weighted_stats(data)

    if args.output:
        result.to_csv(args.output, index=False)
    else:
        print(result.to_csv(index=False))


if __name__ == '__main__':
    main()
