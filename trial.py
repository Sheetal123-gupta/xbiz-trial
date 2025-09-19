import cv2
import numpy as np
import pytesseract
from PIL import Image
import os

class OCRTextExtractor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise ValueError(f"Could not load image from path: {image_path}")
        
        print(f"✓ Image loaded successfully: {image_path}")
        print(f"✓ Image dimensions: {self.original_image.shape}")
        
    def level1_basic_enhancement(self, image):
        """Level 1: Basic Enhancement with debug output"""
        print("Running Level 1 Basic Enhancement...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        print(f"  - Converted to grayscale: {gray.shape}")
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        print("  - Applied CLAHE")
        
        # Gaussian blur + unsharp masking
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        unsharp_mask = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        print("  - Applied unsharp masking")
        
        # Adaptive thresholding
        binary = cv2.adaptiveThreshold(unsharp_mask, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        print("  - Applied adaptive thresholding")
        
        return binary
    
    def extract_text_with_config(self, processed_image, psm_mode=6):
        """Extract text using Tesseract with specific configuration"""
        try:
            print(f"  - Running OCR with PSM mode {psm_mode}")
            
            # Simple configuration first
            text = pytesseract.image_to_string(processed_image, lang='eng')
            print(f"  - Extracted {len(text)} characters")
            
            # Try to get confidence (this might fail sometimes)
            try:
                data = pytesseract.image_to_data(processed_image, output_type=pytesseract.Output.DICT)
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            except:
                avg_confidence = 50.0  # Default confidence if calculation fails
            
            return text.strip(), avg_confidence
            
        except Exception as e:
            print(f"  - OCR failed: {e}")
            return "", 0

    def simple_extraction_test(self):
        """Simple test to see if basic processing works"""
        print("\n" + "="*50)
        print("RUNNING SIMPLE EXTRACTION TEST")
        print("="*50)
        
        results = []
        
        # Test 1: Direct OCR on original image
        try:
            print("\nTest 1: Direct OCR on original image")
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            text, conf = self.extract_text_with_config(gray)
            results.append(("Direct OCR", text, conf))
            print(f"Result: {len(text)} characters extracted")
        except Exception as e:
            print(f"Test 1 failed: {e}")
        
        # Test 2: Simple thresholding
        try:
            print("\nTest 2: Simple binary thresholding")
            gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text, conf = self.extract_text_with_config(binary)
            results.append(("Binary Threshold", text, conf))
            print(f"Result: {len(text)} characters extracted")
        except Exception as e:
            print(f"Test 2 failed: {e}")
        
        # Test 3: Level 1 enhancement
        try:
            print("\nTest 3: Level 1 Enhancement")
            enhanced = self.level1_basic_enhancement(self.original_image)
            text, conf = self.extract_text_with_config(enhanced)
            results.append(("Level 1 Enhanced", text, conf))
            print(f"Result: {len(text)} characters extracted")
        except Exception as e:
            print(f"Test 3 failed: {e}")
        
        return results

# Simple usage
def main():
    try:
        print("Starting OCR Text Extraction...")
        extractor = OCRTextExtractor('check.jpg')
        
        # Run simple tests first
        results = extractor.simple_extraction_test()
        
        # Display all results
        print("\n" + "="*50)
        print("FINAL RESULTS")
        print("="*50)
        
        for method, text, confidence in results:
            print(f"\nMethod: {method}")
            print(f"Confidence: {confidence:.2f}%")
            print(f"Text Length: {len(text)} characters")
            if text:
                print("First 200 characters:")
                print(repr(text[:200]))  # Using repr to see whitespace/newlines
            else:
                print("No text extracted")
            print("-" * 40)
        
        # Find best result
        if results:
            best = max(results, key=lambda x: len(x[1]))  # Best by text length
            print(f"\nBest result by text length: {best[0]}")
            print(f"Full text:\n{best[1]}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()