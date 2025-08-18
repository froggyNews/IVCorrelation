# Ensuring `termplot()` Uses a Full Plot Region

When `termplot()` is drawn after other plots that set a multi-panel layout (for example using `par(mfrow = c(2, 2))`), the layout settings remain active. As a result, `termplot()` is rendered in one of the subregions instead of occupying the full plotting device.

To restore a full-size plotting region for `termplot()` while keeping prior layout settings intact, temporarily reset the graphics parameters:

```r
op <- par(no.readonly = TRUE)  # save current graphics settings
par(mfrow = c(1, 1))          # reset to a single plotting region
termplot(fit)                 # draw the term plot
par(op)                       # restore previous settings
```

### Alternatives

- **Open a new graphics device** before calling `termplot()` (e.g., `dev.new()`) and close it afterward.
- **Use a helper function** that resets and restores the layout automatically, such as wrapping the code above in a function and relying on `on.exit(par(op))`.

These approaches allow `termplot()` to display at full size without permanently altering the layout for subsequent plots.
