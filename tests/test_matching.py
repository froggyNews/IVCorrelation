#!/usr/bin/env python3
"""Test improved legend label matching logic"""

series_keys = ['Observed Points', 'SVI Fit', 'SVI Confidence Interval']
legend_labels = ['Observed', 'SVI fit', '68% CI']

print('Testing improved label matching logic...')
for legend_label in legend_labels:
    matched = None
    # Try exact match first, then partial match
    for key in series_keys:
        if key == legend_label or legend_label in key or key in legend_label:
            matched = key
            break
    
    if not matched:
        # Fallback: try to match based on common words or special cases
        label_words = legend_label.lower().split()
        for key in series_keys:
            key_words = key.lower().split()
            # Check for word overlap
            if any(word in key_words for word in label_words):
                matched = key
                break
            # Special case: CI/Confidence Interval matching
            if ('ci' in legend_label.lower() or '%' in legend_label) and 'confidence' in key.lower():
                matched = key
                break
            # Special case: fit matching
            if 'fit' in legend_label.lower() and 'fit' in key.lower():
                matched = key
                break
    
    print(f'  "{legend_label}" -> "{matched}"')

print('\nAll legend labels should now match correctly!')
