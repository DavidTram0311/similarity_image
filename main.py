from image_analyzer_v2 import ImageAnalyzerv2

# Initialize with reference images
analyzer = ImageAnalyzerv2(
    image_ref1='images/input1.png',
    image_ref2='images/input2.png',
)

# Process an input image
color_name, similarity = analyzer.process('images/red.jpg')
print(f"Dominant color: {color_name}")
print(f"Similarity score: {str(similarity)}%")