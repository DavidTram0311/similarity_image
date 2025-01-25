from image_analyzer_v2 import ImageAnalyzerv2
"""
This script is apply booling cursor to the image
Red: 0
Blue: 1
Others: 2
"""
# Initialize with reference images
analyzer = ImageAnalyzerv2(min_similarity=0.5, image_ref1='images/blue_cursor.png', image_ref2='images/red_cursor.png')

crop = analyzer.booling_cursor(image_path='images/red_diff_input.png', margin_size = 0)
print(crop)


