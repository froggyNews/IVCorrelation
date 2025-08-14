# Animation Integration for IVCorrelation Browser

This document describes the new animation features integrated into the IV Correlation browser interface, providing smooth transitions instead of full plot reloads.

## Overview

The animation integration extends the existing browser GUI (`display/gui/browser.py`) with animation capabilities for supported plot types. Instead of static plot refreshes, users can now view animated transitions that show how volatility surfaces and smile curves evolve over time or across different expiries.

## Features

### Animated Plot Types

1. **Smile Plots** (`Smile (K/S vs IV)`)
   - Animates over multiple dates (time series) when available
   - Falls back to animating across expiries for single dates  
   - Shows smooth transitions in volatility smile shape
   - Provides insights into volatility evolution

2. **Synthetic Surface Plots** (`Synthetic Surface (Smile)`)
   - Animates 3D volatility surfaces over time
   - Shows how the entire IV surface changes across dates
   - Useful for understanding market dynamics

### GUI Controls

The browser now includes animation control buttons:

- **Animate Checkbox**: Enable/disable animation mode for supported plot types
- **Play/Pause Button**: Control animation playback
- **Stop Button**: Stop and reset animation
- **Speed Slider**: Adjust animation speed (100ms to 2000ms between frames)

### Fallback Behavior

- Animation is only available for supported plot types
- Automatically falls back to static plots when:
  - Plot type doesn't support animation
  - Insufficient data for animation (< 2 data points)
  - Animation creation fails
- All existing functionality remains unchanged

## Technical Implementation

### PlotManager Extensions

The `PlotManager` class (`display/gui/gui_plot_manager.py`) has been extended with:

```python
# Animation state management
_animation: FuncAnimation | None
_animation_paused: bool
_animation_speed: int

# Animation control methods  
def has_animation_support(plot_type: str) -> bool
def is_animation_active() -> bool
def start_animation() -> bool
def pause_animation() -> bool  
def stop_animation()
def set_animation_speed(interval_ms: int)
def plot_animated(ax, settings) -> bool
```

### Browser GUI Extensions

The `BrowserApp` class (`display/gui/browser.py`) includes:

- New animation control widgets in the navigation bar
- Updated `_refresh_plot()` method to support animated plotting
- Animation button state management
- Integration with existing plot workflow

### Animation Data Processing

For **Smile animations**:
1. Attempts to load multiple dates for the target ticker
2. Falls back to multiple expiries if insufficient date data
3. Creates common moneyness grid for smooth interpolation
4. Generates smooth transitions between frames

For **Surface animations**:
1. Loads surface grids across multiple dates
2. Aligns surfaces to common grid structure
3. Creates animated heatmap transitions

## Usage

### Starting Animation

1. Select a supported plot type (Smile or Synthetic Surface)
2. Check the "Animate" checkbox
3. Click "Plot" to generate the animated visualization
4. Use Play/Pause/Stop controls as needed
5. Adjust speed with the speed slider

### Example Workflow

```python
# The animation system works automatically:
# 1. User selects "Smile (K/S vs IV)" plot type
# 2. User checks "Animate" checkbox  
# 3. User clicks "Plot"
# 4. System attempts animated plotting:
#    - Loads multiple dates for target ticker
#    - Creates animation frames
#    - Displays smooth transitions
# 5. If animation fails, falls back to static plot
```

## Integration with Existing Animation Utilities

The integration leverages existing animation utilities in `src/viz/anim_utils.py`:

- `animate_smile_over_time()`: For smile curve animations
- `animate_surface_timesweep()`: For surface animations  
- Interactive controls: checkboxes, keyboard toggles, legend toggles

## Backward Compatibility

- All existing functionality is preserved
- Animation is purely additive - no breaking changes
- Static plotting remains the default behavior
- GUI gracefully handles animation failures

## Performance Considerations

- Animations are created on-demand only when requested
- Limited to reasonable number of frames (typically 8-10 dates)
- Uses efficient matplotlib FuncAnimation with blitting
- Smooth interpolation on common grids for consistent performance

## Testing

Comprehensive tests are included:

- `test_animation_integration.py`: Core animation functionality
- `test_browser_integration.py`: Browser integration without GUI
- `demo_animation.py`: Visual demonstration with synthetic data

## Future Enhancements

Potential future improvements:

1. **Additional Plot Types**: Extend animation to Term Structure plots
2. **Export Capabilities**: Save animations as MP4 or GIF files
3. **Interactive Controls**: Mouse-over frame selection
4. **Custom Speed Curves**: Non-linear animation timing
5. **Multi-Ticker Comparisons**: Animate multiple tickers simultaneously

## Troubleshooting

### Animation Not Available
- Ensure plot type supports animation (Smile or Synthetic Surface)
- Verify sufficient data points (minimum 2 dates/expiries)
- Check that animation checkbox is enabled

### Performance Issues  
- Reduce animation speed (increase interval)
- Limit number of animation frames
- Use smaller data sets for testing

### GUI Issues
- Animation controls are disabled for unsupported plot types
- Button states update automatically based on animation status
- All controls gracefully handle edge cases

## Code Examples

### Basic Animation Usage
```python
# In PlotManager
if settings["plot_type"].startswith("Smile"):
    if self.plot_animated(ax, settings):
        print("Animation created successfully")
    else:
        print("Falling back to static plot")
        self.plot(ax, settings)
```

### Custom Animation Speed
```python
mgr = PlotManager()
mgr.set_animation_speed(1000)  # 1 second between frames
```

### Animation State Checking
```python
if mgr.is_animation_active():
    mgr.pause_animation()
else:
    print("No animation running")
```

This animation integration provides a more engaging and insightful visualization experience while maintaining full backward compatibility with existing functionality.