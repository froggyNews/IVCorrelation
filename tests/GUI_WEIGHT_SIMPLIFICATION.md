# GUI Weight Settings Simplification Plan

## Current Problems
1. **8 weight modes** in GUI dropdown, but **4 are broken** (PCA modes)
2. **3 overlapping modes** that do the same thing with different names
3. **Missing controls** for `weight_power` and `clip_negative` 
4. **Hardcoded parameters** not exposed to users

## Recommended Simplification

### Phase 1: Remove Broken Options
Remove from GUI dropdown:
- `"pca_atm_market"` ❌ 
- `"pca_atm_regress"` ❌
- `"pca_surface_market"` ❌ 
- `"pca_surface_regress"` ❌

### Phase 2: Consolidate Overlapping Options
Replace current modes:
```python
# OLD (confusing)
"iv_atm"    → "corr"       # Correlation weights using IV ATM
"surface"   → "corr_surf"  # Correlation weights using IV surface  
"ul"        → "corr_ul"    # Correlation weights using underlying

# NEW (clear)
Weight Mode: ["corr", "equal", "custom", "surface_grid"]
Feature Set: ["atm", "surface", "underlying"]  # Only visible when corr selected
```

### Phase 3: Add Missing Controls
```python
# Add these controls to GUI:
Weight Power:     [Slider: 1.0 - 3.0, default=1.0]
Clip Negative:    [Checkbox: default=True] 
Advanced Modes:   [Checkbox: "Enable cosine/experimental modes"]
```

### Phase 4: Final Simplified GUI Layout
```
┌─ Weight Settings ────────────────────────────┐
│ Mode: [corr ▼] Feature: [atm ▼]              │
│ Power: [1.0 ████▒▒▒] Clip Negative: ☑       │
│ ☐ Advanced (cosine, PCA when fixed)         │
└──────────────────────────────────────────────┘
```

## Benefits
✅ **Removes 4 broken options** (50% reduction in GUI complexity)  
✅ **Clarifies overlapping functionality** (mode vs feature separation)
✅ **Exposes hidden parameters** (power, clipping)
✅ **Maintains backward compatibility** (old modes still work in code)
✅ **Future-ready** (easy to add cosine, fixed PCA later)

## Implementation
1. Update `gui_input.py` combobox values
2. Add power slider and clip checkbox  
3. Add conditional feature dropdown
4. Update `_sync_settings()` to handle new controls
5. Update `compute_peer_weights()` parameter mapping
