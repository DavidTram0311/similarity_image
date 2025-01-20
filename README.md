# Image Analyzer

A high-performance image analysis tool specialized in color detection and image comparison, with a focus on accuracy and real-time processing.

## Features

- Dominant color detection with color name mapping
- Multi-metric image comparison (SSIM, histogram, contour)
- Optimized for real-time processing using parallel execution
- Support for multiple reference images
- Automatic image resizing for large inputs

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- numpy >= 1.21.0
- opencv-python >= 4.5.0
- scipy >= 1.7.0
- scikit-image >= 0.18.0
- webcolors == 1.11

## Usage

### Basic Usage

```python
from image_analyzer import ImageAnalyzer

# Initialize with reference images
analyzer = ImageAnalyzer(
    min_similarity=0.95,
    image_ref1='path/to/reference1.jpg',
    image_ref2='path/to/reference2.jpg'
)

# Process an input image
color_name, similarity = analyzer.process('path/to/input.jpg')
print(f"Dominant color: {color_name}")
print(f"Similarity score: {similarity:.2f}%")
```

### Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)

## Technical Details

### Image Comparison Method

The tool uses a weighted combination of three comparison metrics:
- Structural Similarity Index (SSIM) - 40% weight
- Color Histogram Comparison - 30% weight
- Contour Matching - 30% weight

### Color Detection

- Uses HSV color space for white pixel detection
- Implements KDTree for efficient color name mapping
- Supports all CSS3 color names
- Caches color name lookups for better performance

### White Color Detection Parameters

- Saturation maximum: 30
- Value minimum: 200

## Performance Optimizations

- Parallel processing for image loading and analysis
- LRU cache for color name lookup
- Automatic image resizing for large inputs
- ThreadPoolExecutor for concurrent operations

## Error Handling

The tool includes comprehensive error checking for:
- Invalid file paths
- Unsupported image formats
- Image loading failures
- Size mismatches between images
- Color matching failures

## Limitations

- Input and reference images should have matching dimensions
- Optimized for images with two main colors (white and another dominant color)
- Requires high similarity (default threshold: 0.95) for positive matches