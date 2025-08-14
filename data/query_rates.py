#!/usr/bin/env python3
"""
Utility to query and display ticker-specific interest rates.
"""
import sys
import os
import pandas as pd

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.interest_rates import (
    get_ticker_interest_rate, 
    list_tickers_with_rates, 
    get_ticker_rate_history,
    get_most_recent_ticker_rates_date
)

def query_ticker_rates():
    """Interactive tool to query ticker rates."""
    print("=== Ticker Interest Rate Query Tool ===")
    print(f"Most recent data date: {get_most_recent_ticker_rates_date()}")
    
    while True:
        print("\nOptions:")
        print("1. Get rate for specific ticker")
        print("2. Show summary statistics")
        print("3. Show top 10 highest rates")
        print("4. Show top 10 lowest rates (most negative)")
        print("5. Search tickers by partial name")
        print("6. Show GC vs HTB breakdown")
        print("7. Exit")
        
        choice = input("\nEnter choice (1-7): ").strip()
        
        if choice == '1':
            ticker = input("Enter ticker symbol: ").strip().upper()
            if ticker:
                rate = get_ticker_interest_rate(ticker)
                history = get_ticker_rate_history(ticker)
                
                if rate is not None:
                    print(f"\n{ticker}: {rate:.4f}%")
                    if len(history) > 1:
                        print("Rate history:")
                        for date, rate_val, status in history[:5]:
                            print(f"  {date}: {rate_val:.4f}% ({status or 'N/A'})")
                else:
                    print(f"\n{ticker}: No specific rate found (using default)")
        
        elif choice == '2':
            all_rates = list_tickers_with_rates()
            if all_rates:
                rates_values = [rate for _, rate, _ in all_rates]
                print(f"\nSummary Statistics:")
                print(f"  Total tickers: {len(all_rates)}")
                print(f"  Rate range: {min(rates_values):.4f}% to {max(rates_values):.4f}%")
                print(f"  Mean rate: {sum(rates_values)/len(rates_values):.4f}%")
                print(f"  Median rate: {sorted(rates_values)[len(rates_values)//2]:.4f}%")
        
        elif choice == '3':
            all_rates = list_tickers_with_rates()
            sorted_rates = sorted(all_rates, key=lambda x: x[1], reverse=True)
            print(f"\nTop 10 Highest Rates:")
            for i, (ticker, rate, status) in enumerate(sorted_rates[:10]):
                print(f"  {i+1:2d}. {ticker:6s}: {rate:8.4f}% ({status or 'N/A'})")
        
        elif choice == '4':
            all_rates = list_tickers_with_rates()
            sorted_rates = sorted(all_rates, key=lambda x: x[1])
            print(f"\nTop 10 Lowest Rates (Most Negative):")
            for i, (ticker, rate, status) in enumerate(sorted_rates[:10]):
                print(f"  {i+1:2d}. {ticker:6s}: {rate:8.4f}% ({status or 'N/A'})")
        
        elif choice == '5':
            search = input("Enter partial ticker name: ").strip().upper()
            if search:
                all_rates = list_tickers_with_rates()
                matches = [(ticker, rate, status) for ticker, rate, status in all_rates if search in ticker]
                if matches:
                    print(f"\nFound {len(matches)} matches:")
                    for ticker, rate, status in sorted(matches)[:20]:  # Show first 20
                        print(f"  {ticker:6s}: {rate:8.4f}% ({status or 'N/A'})")
                    if len(matches) > 20:
                        print(f"  ... and {len(matches) - 20} more")
                else:
                    print(f"\nNo tickers found containing '{search}'")
        
        elif choice == '6':
            all_rates = list_tickers_with_rates()
            gc_rates = [rate for _, rate, status in all_rates if status == 'GC']
            htb_rates = [rate for _, rate, status in all_rates if status == 'HTB']
            other_rates = [rate for _, rate, status in all_rates if status not in ['GC', 'HTB']]
            
            print(f"\nBorrow Status Breakdown:")
            print(f"  GC (General Collateral): {len(gc_rates)} tickers")
            if gc_rates:
                print(f"    Rate range: {min(gc_rates):.4f}% to {max(gc_rates):.4f}%")
                print(f"    Mean rate: {sum(gc_rates)/len(gc_rates):.4f}%")
            
            print(f"  HTB (Hard to Borrow): {len(htb_rates)} tickers")
            if htb_rates:
                print(f"    Rate range: {min(htb_rates):.4f}% to {max(htb_rates):.4f}%")
                print(f"    Mean rate: {sum(htb_rates)/len(htb_rates):.4f}%")
            
            if other_rates:
                print(f"  Other/Unknown: {len(other_rates)} tickers")
                print(f"    Mean rate: {sum(other_rates)/len(other_rates):.4f}%")
        
        elif choice == '7':
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice. Please enter 1-7.")

if __name__ == "__main__":
    query_ticker_rates()
