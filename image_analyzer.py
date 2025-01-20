import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Union, Optional, Dict
from scipy.spatial import KDTree
from webcolors import CSS3_NAMES_TO_HEX, hex_to_rgb
from skimage.metrics import structural_similarity as ssim
import concurrent.futures
from functools import lru_cache

class ImageAnalyzer:
    """
    A class for analyzing and comparing images, specifically designed for:
    - Detecting dominant colors in images
    - Comparing similarity between input and reference images
    - Real-time processing with high accuracy
    
    The class assumes that input and reference images:
    - Have the same dimensions
    - Primarily contain two colors (white and another dominant color)
    - Require 100% accuracy in comparison
    """
    
    def __init__(self, min_similarity: float = 0.95, image_ref1: Union[str, Path] = '', image_ref2: Union[str, Path] = ''):
        """
        Initialize the ImageAnalyzer with configuration parameters.
        
        Args:
            min_similarity (float): Minimum similarity threshold (0-1).
                                  Default is 0.95 for high accuracy requirements.
        """
        self.min_similarity = min_similarity
        self.image_ref1 = image_ref1
        self.image_ref2 = image_ref2
        # Khởi tạo color mapping cho việc chuyển đổi RGB sang tên màu
        self._initialize_color_mapping()
        # Định nghĩa ngưỡng cho pixel trắng trong không gian HSV
        self.white_threshold = {
            'saturation_max': 30,  # Ngưỡng độ bão hòa tối đa cho màu trắng
            'value_min': 200       # Ngưỡng giá trị tối thiểu cho màu trắng
        }
        # Pre-compute thresholds for better performance
        self._compute_thresholds()

    def _compute_thresholds(self):
        """
        Pre-compute thresholds và masks cho việc tối ưu hóa
        """
        # Tạo mask cho việc phát hiện màu trắng trong HSV space
        s_max = self.white_threshold['saturation_max']
        v_min = self.white_threshold['value_min']
        
        # Pre-compute ranges cho việc phát hiện màu trắng
        self.white_lower = np.array([0, 0, v_min])
        self.white_upper = np.array([180, s_max, 255])

    @lru_cache(maxsize=128)
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Cached version của hàm chuyển đổi RGB sang tên màu.
        
        Args:
            rgb_color: Tuple RGB (r, g, b)
            
        Returns:
            str: Tên màu tiếng Anh
        """
        distance, index = self.color_tree.query(rgb_color)
        return self.color_names[index]
    
    def get_dominant_color(self, image: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Optimized version của hàm lấy màu chủ đạo.
        
        Args:
            image: Ảnh BGR numpy array
            
        Returns:
            Tuple[np.ndarray, str]: (Giá trị màu BGR, tên màu)
        """
        # Chuyển sang HSV và tìm pixels không phải màu trắng
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        non_white_mask = cv2.inRange(hsv_image, self.white_lower, self.white_upper)
        non_white_mask = cv2.bitwise_not(non_white_mask)
        
        if cv2.countNonZero(non_white_mask) == 0:
            raise ValueError("No dominant color found - image appears to be all white")
            
        # Tính trung bình màu của các pixel không phải màu trắng
        dominant_color = cv2.mean(image, mask=non_white_mask)[:3]
        dominant_color = np.uint8([[dominant_color]])
        
        # Chuyển sang RGB để lấy tên màu
        rgb_color = cv2.cvtColor(dominant_color, cv2.COLOR_BGR2RGB)[0][0]
        color_name = self._get_color_name(tuple(rgb_color))
        
        return dominant_color[0], color_name
    
    def _parallel_histogram_comparison(self,
                                    hsv1: np.ndarray,
                                    hsv2: np.ndarray,
                                    channel: int,
                                    bins: int = 32) -> float:
        """
        So sánh histogram cho một kênh màu.
        
        Args:
            hsv1, hsv2: Ảnh HSV
            channel: Kênh màu (0=H, 1=S, 2=V)
            bins: Số lượng bins
            
        Returns:
            float: Điểm tương đồng histogram
        """
        hist1 = cv2.calcHist([hsv1], [channel], None, [bins], [0, 256])
        hist2 = cv2.calcHist([hsv2], [channel], None, [bins], [0, 256])
        
        cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
        cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _compare_color_histograms(self,
                                img1: np.ndarray,
                                img2: np.ndarray,
                                bins: int = 32) -> float:
        """
        So sánh histogram màu song song.
        """
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for channel in range(3):
                future = executor.submit(
                    self._parallel_histogram_comparison,
                    hsv1, hsv2, channel, bins
                )
                futures.append(future)
                
            hist_scores = [future.result() for future in futures]
            
        return np.mean(hist_scores)
    
    @staticmethod
    def resize_if_needed(image: np.ndarray, 
                        max_dimension: int = 800) -> np.ndarray:
        """
        Resize ảnh nếu kích thước quá lớn.
        
        Args:
            image: Ảnh input
            max_dimension: Kích thước tối đa cho phép
            
        Returns:
            np.ndarray: Ảnh đã resize nếu cần
        """
        height, width = image.shape[:2]
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image

    # Fix the color mapping initialization
    def _initialize_color_mapping(self) -> None:
        """
        Initialize color mapping to convert RGB to color names.
        Uses KDTree for efficient nearest color lookup.
        """
        # Convert CSS3 colors to RGB
        rgb_colors = []
        self.color_names = []
        
        for name, hex_color in CSS3_NAMES_TO_HEX.items():
            rgb = hex_to_rgb(hex_color)
            rgb_colors.append(rgb)
            self.color_names.append(name)
            
        # Create KDTree for nearest color lookup
        self.color_tree = KDTree(rgb_colors)
        
    def _get_color_name(self, rgb_color: Tuple[int, int, int]) -> str:
        """
        Chuyển đổi giá trị RGB sang tên màu gần nhất.
        
        Args:
            rgb_color: Tuple RGB (r, g, b)
            
        Returns:
            str: Tên màu tiếng Anh
        """
        distance, index = self.color_tree.query(rgb_color)
        return self.color_names[index]
    
    def _is_white_pixel(self, pixel: np.ndarray) -> bool:
        """
        Kiểm tra xem một pixel HSV có phải là màu trắng không.
        
        Args:
            pixel: Mảng numpy chứa giá trị HSV của pixel
            
        Returns:
            bool: True nếu pixel được coi là màu trắng
        """
        # H: Hue, S: Saturation, V: Value
        h, s, v = pixel
        return s <= self.white_threshold['saturation_max'] and v >= self.white_threshold['value_min']
    
    def _compare_contours(self,
                         gray1: np.ndarray,
                         gray2: np.ndarray,
                         threshold: int = 127) -> float:
        """
        So sánh hình dạng đối tượng sử dụng contours.
        
        Args:
            gray1, gray2: Ảnh grayscale
            threshold: Ngưỡng nhị phân hóa
            
        Returns:
            float: Điểm tương đồng contour (0-1)
        """
        # Nhị phân hóa ảnh
        _, bin1 = cv2.threshold(gray1, threshold, 255, cv2.THRESH_BINARY)
        _, bin2 = cv2.threshold(gray2, threshold, 255, cv2.THRESH_BINARY)
        
        # Tìm contours
        contours1, _ = cv2.findContours(bin1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(bin2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours1 or not contours2:
            return 0.0
            
        # Lấy contour lớn nhất
        c1 = max(contours1, key=cv2.contourArea)
        c2 = max(contours2, key=cv2.contourArea)
        
        # So sánh hình dạng
        match_score = cv2.matchShapes(c1, c2, cv2.CONTOURS_MATCH_I2, 0.0)
        
        # Chuyển điểm sang thang 0-1
        similarity = 1.0 / (1.0 + match_score)
        
        return similarity
    
    def compare_images(self, 
                      input_image: np.ndarray, 
                      reference_image: np.ndarray
                     ) -> float:
        """
        So sánh hai ảnh sử dụng nhiều phương pháp để đạt độ chính xác cao nhất.
        
        Args:
            input_image: Ảnh input dạng BGR
            reference_image: Ảnh tham khảo dạng BGR
            
        Returns:
            float: Điểm tương đồng (0-1)
        """
        # Validate kích thước ảnh
        # self._validate_images_size(input_image, reference_image)
        input_image = cv2.resize(input_image, (reference_image.shape[1], reference_image.shape[0]))
        
        # Chuyển ảnh sang grayscale cho việc so sánh cấu trúc
        gray1 = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        
        # 1. Tính toán điểm SSIM
        ssim_score, _ = ssim(gray1, gray2, full=True)
        
        # 2. So sánh histogram màu
        hist_score = self._compare_color_histograms(input_image, reference_image)
        
        # 3. So sánh contour
        contour_score = self._compare_contours(gray1, gray2)
        
        # Tính điểm tổng hợp (có thể điều chỉnh trọng số)
        weights = {
            'ssim': 0.4,
            'histogram': 0.3,
            'contour': 0.3
        }
        
        final_score = (
            ssim_score * weights['ssim'] +
            hist_score * weights['histogram'] +
            contour_score * weights['contour']
        )
        
        return final_score

    def _validate_image_path(self, image_path: Union[str, Path]) -> bool:
        """
        Validate if the image path exists and has a supported format.
        
        Args:
            image_path: Path to the image file (string or Path object)
            
        Returns:
            bool: True if valid, False otherwise
            
        Raises:
            ValueError: If image format is not supported
        """
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp'}
        path = Path(image_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
            
        if path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats are: {supported_formats}"
            )
            
        return True
        
    def _load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and validate an image file.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Loaded image in BGR format
            
        Raises:
            ValueError: If image cannot be loaded
        """
        self._validate_image_path(image_path)
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        return image
        
    def _validate_images_size(self, image1: np.ndarray, image2: np.ndarray) -> bool:
        """
        Validate if two images have the same dimensions.
        
        Args:
            image1: First image array
            image2: Second image array
            
        Returns:
            bool: True if images have same dimensions
            
        Raises:
            ValueError: If images have different dimensions
        """
        if image1.shape != image2.shape:
            raise ValueError(
                f"Images have different dimensions: "
                f"{image1.shape} vs {image2.shape}"
            )
        return True

    def extract_basic_color(dominant_color_name: str) -> str:
        """
        Extract the basic color name from the dominant color name.
        
        Args:
            dominant_color_name: Dominant color name
            
        Returns:
            str: Basic color name
        """
        basic_colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
        for color in basic_colors:
            if color in dominant_color_name:
                return color
        return dominant_color_name

    def process(self, 
                input_path: Union[str, Path], 
               ) -> Tuple[str, float]:
        """
        Optimized version của hàm xử lý chính.
        """
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Đọc ảnh song song
            future_input = executor.submit(self._load_image, input_path)

            future_reference1 = executor.submit(self._load_image, self.image_ref1)
            future_reference2 = executor.submit(self._load_image, self.image_ref2)

            input_image = future_input.result()
            reference_image1 = future_reference1.result()
            reference_image2 = future_reference2.result()
            
            # Xử lý song song màu chủ đạo và so sánh ảnh
            future_input_color = executor.submit(self.get_dominant_color, input_image)
            future_ref1_color = executor.submit(self.get_dominant_color, reference_image1)
            future_ref2_color = executor.submit(self.get_dominant_color, reference_image2)

            _, dominant_input_color_name = future_input_color.result()
            _, dominant_ref1_color_name = future_ref1_color.result()
            _, dominant_ref2_color_name = future_ref2_color.result()
            
            print(f"Input image dominant color: {dominant_input_color_name}")
            print(f"Reference image 1 dominant color: {dominant_ref1_color_name}")
            print(f"Reference image 2 dominant color: {dominant_ref2_color_name}")

            basic_colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", "gray"]
            for color in basic_colors:
                if color in dominant_input_color_name:
                    dominant_input_color_name = color
                if color in dominant_ref1_color_name:
                    dominant_ref1_color_name = color
                if color in dominant_ref2_color_name:
                    dominant_ref2_color_name = color

            if dominant_input_color_name == dominant_ref1_color_name:
                reference_image = reference_image1
            elif dominant_input_color_name == dominant_ref2_color_name:
                reference_image = reference_image2
            else:
                raise ValueError("Input image color does not match any reference image color")
            

            future_similarity = executor.submit(
                self.compare_images, input_image, reference_image
            )
            
            similarity = future_similarity.result()
            
        similarity_percentage = similarity * 100
        return dominant_input_color_name, similarity_percentage