```markdown
# CRTools: Cosmic Ray Removal Tools

Utility functions for removing cosmic ray hits in single exposures,
implemented as C extensions.

## Overview

CRTools implements two core algorithms for cosmic ray detection and removal:

- **`fmedian`**: Local median computation
- **`fsigma`**: Local standard deviation calculation

Both functions are implemented in C for maximum performance while maintaining a simple Python interface.

## Features

- **Flexible Window Sizes**: Configurable neighborhood sizes for different image characteristics
- **NumPy Integration**: Seamless integration with NumPy arrays


## Quick Start

```python
import numpy as np
from crtools import fmedian, fsigma

# Create sample astronomical image data
image = np.random.normal(100, 10, (256, 256)).astype(np.float64)
# Add some cosmic ray hits (bright outliers)
image[100, 100] = 5000  # cosmic ray hit

# Method 1: Use filtered median to smooth outliers
smoothed = np.zeros_like(image)
fmedian(image, smoothed, xsize=1, ysize=1, exclude_center=1)

# Method 2: Use sigma-clipping approach
sigma_map = np.zeros_like(image)
fsigma(image, sigma_map, xsize=2, ysize=2, exclude_center=1)

# Calculate z-scores for outlier detection
mean_image = smoothed  # or compute local mean separately
z_scores = (image - mean_image) / (sigma_map + 1e-8)

# Identify cosmic rays (e.g., >5 sigma outliers)
cosmic_ray_mask = np.abs(z_scores) > 5.0

# Replace cosmic rays with smoothed values
cleaned_image = image.copy()
cleaned_image[cosmic_ray_mask] = smoothed[cosmic_ray_mask]
```

## API Reference

### fmedian

Computes a filtered median over a local neighborhood around each pixel.

```python
fmedian(input_array, output_array, xsize, ysize, exclude_center)
```

**Parameters:**

- `input_array` (numpy.ndarray): Input image array (float64)
- `output_array` (numpy.ndarray): Output array for results (float64, same shape as input)
- `xsize` (int): Half-width of window in x-direction
- `ysize` (int): Half-width of window in y-direction  
- `exclude_center` (int): If 1, exclude center pixel from median calculation; if 0, include it

**Window Size:** The actual window size is `(2*xsize+1) × (2*ysize+1)`.

### fsigma

Computes the local standard deviation over a neighborhood around each pixel.

```python
fsigma(input_array, output_array, xsize, ysize, exclude_center)
```

**Parameters:**

- `input_array` (numpy.ndarray): Input image array (float64)
- `output_array` (numpy.ndarray): Output array for standard deviation values (float64, same shape as input)
- `xsize` (int): Half-width of window in x-direction
- `ysize` (int): Half-width of window in y-direction
- `exclude_center` (int): If 1, exclude center pixel from sigma calculation; if 0, include it

## Performance

The C implementations provide significant performance improvements over pure Python/NumPy implementations:

- **fmedian**: Optimized median calculation with boundary handling
- **fsigma**: Efficient standard deviation computation
- **Memory efficient**: In-place operations where possible
- **Boundary handling**: Proper edge/corner pixel treatment

## Editor integration: refresh VS Code C/C++ includes

If you use VS Code and the C/C++ extension (clangd), the editor needs to know
where to find `Python.h` and NumPy headers. This repo includes a helper script
to regenerate the VS Code configuration from the active Python environment.

To refresh the include paths (run in the repository root):

```bash
python3 .vscode/update_includes.py
```

This writes `.vscode/c_cpp_properties.json` with the Python and NumPy include
directories from the interpreter used to run the script. Re-run it after you
switch virtual environments.

```
# CRTools: Cosmic Ray Removal Tools

Utility functions for removing cosmic ray hits in single exposures, 
implemented as C extensions.

## Overview

CRTools implements two core algorithms for cosmic ray detection and removal:

- **`fmedian`**: Local median computation
- **`fsigma`**: Local standard deviation calculation

Both functions are implemented in C for maximum performance while maintaining a simple Python interface.

## Features

- **Flexible Window Sizes**: Configurable neighborhood sizes for different image characteristics
- **NumPy Integration**: Seamless integration with NumPy arrays


## Quick Start

```python
import numpy as np
from crtools import fmedian, fsigma

# Create sample astronomical image data
image = np.random.normal(100, 10, (256, 256)).astype(np.float64)
# Add some cosmic ray hits (bright outliers)
image[100, 100] = 5000  # cosmic ray hit

# Method 1: Use filtered median to smooth outliers
smoothed = np.zeros_like(image)
fmedian(image, smoothed, xsize=1, ysize=1, exclude_center=1)

# Method 2: Use sigma-clipping approach
sigma_map = np.zeros_like(image)
fsigma(image, sigma_map, xsize=2, ysize=2, exclude_center=1)

# Calculate z-scores for outlier detection
mean_image = smoothed  # or compute local mean separately
z_scores = (image - mean_image) / (sigma_map + 1e-8)

# Identify cosmic rays (e.g., >5 sigma outliers)
cosmic_ray_mask = np.abs(z_scores) > 5.0

# Replace cosmic rays with smoothed values
cleaned_image = image.copy()
cleaned_image[cosmic_ray_mask] = smoothed[cosmic_ray_mask]
```

## API Reference

### fmedian

Computes a filtered median over a local neighborhood around each pixel.

```python
fmedian(input_array, output_array, xsize, ysize, exclude_center)
```

**Parameters:**

- `input_array` (numpy.ndarray): Input image array (float64)
- `output_array` (numpy.ndarray): Output array for results (float64, same shape as input)
- `xsize` (int): Half-width of window in x-direction
- `ysize` (int): Half-width of window in y-direction  
- `exclude_center` (int): If 1, exclude center pixel from median calculation; if 0, include it

**Window Size:** The actual window size is `(2*xsize+1) × (2*ysize+1)`.

### fsigma

Computes the local standard deviation over a neighborhood around each pixel.

```python
fsigma(input_array, output_array, xsize, ysize, exclude_center)
```

**Parameters:**

- `input_array` (numpy.ndarray): Input image array (float64)
- `output_array` (numpy.ndarray): Output array for standard deviation values (float64, same shape as input)
- `xsize` (int): Half-width of window in x-direction
- `ysize` (int): Half-width of window in y-direction
- `exclude_center` (int): If 1, exclude center pixel from sigma calculation; if 0, include it

## Performance

The C implementations provide significant performance improvements over pure Python/NumPy implementations:

- **fmedian**: Optimized median calculation with boundary handling
- **fsigma**: Efficient standard deviation computation
- **Memory efficient**: In-place operations where possible
- **Boundary handling**: Proper edge/corner pixel treatment

## Editor integration: refresh VS Code C/C++ includes

If you use VS Code and the C/C++ extension (clangd), the editor needs to know
where to find `Python.h` and NumPy headers. This repo includes a helper script
to regenerate the VS Code configuration from the active Python environment.

To refresh the include paths (run in the repository root):

```bash
python3 .vscode/update_includes.py
```

This writes `.vscode/c_cpp_properties.json` with the Python and NumPy include
directories from the interpreter used to run the script. Re-run it after you
switch virtual environments.
