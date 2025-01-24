from image_analyzer import ImageAnalyzer
"""
This script is apply booling cursor to the image
Red: 0
Blue: 1
"""
# Initialize with reference images
analyzer = ImageAnalyzer()

crop = analyzer.booling_cursor(image_path='images/red_diff_input.png', margin_size = 10)
print(crop)


