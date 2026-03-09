import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")

# Lazy import for EasyOCR to avoid loading it if not needed
_easyocr_reader = None

def _get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            print("Loading EasyOCR for Arabic...")
            _easyocr_reader = easyocr.Reader(['ar'], gpu=False, verbose=False)
        except ImportError:
            raise ImportError("EasyOCR is required for Arabic OCR. Install it with: pip install easyocr")
    return _easyocr_reader


class HandwritingRecognizer:
    def __init__(self, en_model="microsoft/trocr-base-handwritten"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # English TrOCR (lazy loaded)
        self._en_model_name = en_model
        self._en_processor = None
        self._en_model = None

    def _get_en_model(self):
        if self._en_model is None:
            print(f"Loading English OCR Model: {self._en_model_name} on {self.device}...")
            self._en_processor = TrOCRProcessor.from_pretrained(self._en_model_name)
            self._en_model = VisionEncoderDecoderModel.from_pretrained(self._en_model_name).to(self.device)
        return self._en_processor, self._en_model

    def segment_lines(self, cv_image):
        """
        Segments a multi-line image into horizontal line segments using projection profiles.
        """
        # Convert to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Binary thresholding (Invert so text is white)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Identify horizontal lines by opening with a wide horizontal kernel.
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
        horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
        # Dilate slightly to cover the full width of the ruled lines
        horiz_lines = cv2.dilate(horiz_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
        
        # Identify vertical lines by opening with a tall vertical kernel.
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
        vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
        # Dilate slightly to cover the full width of the vertical lines (margins)
        vert_lines = cv2.dilate(vert_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
        
        # Combine line masks
        line_mask = cv2.bitwise_or(horiz_lines, vert_lines)
        
        # Cleaned binary: Remove these lines from the binary image for projection profile
        processed = cv2.subtract(binary, line_mask)
        
        # Cleaned RGB: White out lines in the original color image for OCR
        cleaned_cv_image = cv_image.copy()
        cleaned_cv_image[line_mask > 0] = [255, 255, 255]
        
        # Remove small noise and smooth
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_noise)

        # SMEARING STRATEGY: Dilate horizontally to merge words into solid bars
        kernel_smear = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 1))
        smeared = cv2.dilate(processed, kernel_smear, iterations=1)
        
        # Horizontal projection profile on SMEARED image
        h, w = smeared.shape
        left_detect_margin = int(w * 0.15)
        right_detect_margin = int(w * 0.10)
        cropped_for_profile = smeared[:, left_detect_margin:w-right_detect_margin]
        
        horizontal_sum = np.sum(cropped_for_profile, axis=1)
        
        # Detect Peaks (Centers of lines)
        peaks = []
        v_nonzero = horizontal_sum[horizontal_sum > 0]
        thresh = np.percentile(v_nonzero, 30) if len(v_nonzero) > 0 else 0
        
        for y in range(10, h - 10):
            if horizontal_sum[y] > thresh and \
               horizontal_sum[y] == np.max(horizontal_sum[y-10:y+11]):
                if not peaks or y - peaks[-1] > 30:
                    peaks.append(y)

        if not peaks:
            return [cleaned_cv_image]

        # Calculate split points (Valleys) between peaks
        splits = [0]
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            valley_y = p1 + np.argmin(horizontal_sum[p1:p2])
            splits.append(valley_y)
        splits.append(h)

        line_segments = []
        for i in range(len(splits) - 1):
            y_start = splits[i]
            y_end = splits[i+1]
            
            sub_profile = horizontal_sum[y_start:y_end]
            text_rows = np.where(sub_profile > thresh * 0.2)[0]
            if len(text_rows) > 0:
                y_s = max(y_start, y_start + min(text_rows) - 5)
                y_e = min(y_end, y_start + max(text_rows) + 5)
                
                line_processed = processed[y_s:y_e, :]
                vert_sum = np.sum(line_processed, axis=0)
                text_cols = np.where(vert_sum > 0)[0]
                
                if len(text_cols) > 0:
                    x_s = max(0, min(text_cols) - 10)
                    x_e = min(w, max(text_cols) + 10)
                    
                    if x_e - x_s > 20 and y_e - y_s > 10:
                        segment = cleaned_cv_image[y_s:y_e, x_s:x_e]
                        line_segments.append(segment)

        return line_segments

    def _extract_arabic(self, cv_image):
        """
        Use EasyOCR to extract Arabic text directly from the full (cleaned) image.
        EasyOCR handles its own internal line segmentation for Arabic.
        """
        reader = _get_easyocr_reader()
        results = reader.readtext(cv_image, detail=0, paragraph=True)
        return " ".join(results)

    def extract_text_from_handwriting(self, image_input, lang="en"):
        """
        image_input: path to image or PIL Image object
        lang: language code ('en' or 'ar')
        returns: extracted text
        """
        if isinstance(image_input, str):
            pil_image = Image.open(image_input).convert("RGB")
        else:
            pil_image = image_input.convert("RGB")

        # Convert PIL to OpenCV (BGR)
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        if lang == "ar":
            # Use EasyOCR for Arabic — much faster on CPU, handles RTL
            print("Using EasyOCR for Arabic handwriting...")
            # Clean the image first (remove ruled lines)
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 1))
            horiz_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horiz_kernel)
            horiz_lines = cv2.dilate(horiz_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))
            vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50))
            vert_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vert_kernel)
            vert_lines = cv2.dilate(vert_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
            line_mask = cv2.bitwise_or(horiz_lines, vert_lines)
            cleaned = cv_image.copy()
            cleaned[line_mask > 0] = [255, 255, 255]
            return self._extract_arabic(cleaned)
        else:
            # Use TrOCR for English
            line_segments = self.segment_lines(cv_image)
            processor, model = self._get_en_model()
            
            full_text = []
            print(f"Detected {len(line_segments)} lines in image. Using English TrOCR.")
            
            for segment in line_segments:
                segment_rgb = cv2.cvtColor(segment, cv2.COLOR_BGR2RGB)
                segment_pil = Image.fromarray(segment_rgb)
                
                pixel_values = processor(images=segment_pil, return_tensors="pt").pixel_values.to(self.device)
                generated_ids = model.generate(pixel_values)
                line_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                if line_text:
                    full_text.append(line_text)
            
            return " ".join(full_text)


def extract_text_from_handwriting(image_path):
    recognizer = HandwritingRecognizer()
    return recognizer.extract_text_from_handwriting(image_path)
