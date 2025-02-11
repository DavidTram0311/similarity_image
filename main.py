from image_analyzer_v2 import ImageAnalyzerv2

# Initialize with reference images
analyzer = ImageAnalyzerv2(
    image_ref1='images/blue_ref_11_2.jpg',
    image_ref2='images/red_ref_11_2.jpg',
)

# Process an input image
color_name, similarity = analyzer.process('images/red_11_2.jpg')
print(f"Dominant color: {color_name}")
print(f"Similarity score: {str(similarity)}%")