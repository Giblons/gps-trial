import cv2
import numpy as np

class Preprocessor:
    """
    Optical Calibration & Advanced Preprocessing
    Implements fisheye undistortion and various thresholding/color space conversions
    to prepare the frame for semantic segmentation and feature matching.
    """
    def __init__(self, K=None, D=None, img_shape=(1920, 1080)):
        if K is None:
            focal_length = 800
            self.K = np.array([[focal_length, 0, img_shape[0]/2],
                               [0, focal_length, img_shape[1]/2],
                               [0, 0, 1]], dtype=np.float32)
        else:
            self.K = np.array(K, dtype=np.float32)

        if D is None:
            self.D = np.array([[-0.1], [0.01], [-0.001], [0.0001]], dtype=np.float32)
        else:
            self.D = np.array(D, dtype=np.float32)

        self.img_shape = img_shape

        # balance=0.0 yields pristine orthogonal ROI without undefined black edges
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistort(
            self.K, self.D, self.img_shape, np.eye(3), balance=0.0
        )

        # Precompute undistortion maps for SIMD performance on ARM NEON
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, self.img_shape, cv2.CV_16SC2
        )

    def undistort(self, frame):
        """Applies fisheye undistortion."""
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def get_binarization_masks(self, gray_frame):
        """
        Dynamic Binarization & Thresholding:
        Returns both Otsu's Global Thresholding (for bimodal illumination)
        and Adaptive Gaussian Thresholding (for handling complex shadows).
        """
        _, otsu = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        adaptive = cv2.adaptiveThreshold(
            gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return otsu, adaptive

    def convert_color_spaces(self, frame):
        """
        Color Space Conversion:
        Converts the live feed into various color spaces (HSV, LAB, GRAY).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        return gray, hsv, lab

    def preprocess_pipeline(self, frame):
        """
        Executes the full SIMD-optimized preprocessing pipeline.
        Returns multiple modalities for the matcher to use.
        """
        undistorted = self.undistort(frame)
        gray, hsv, lab = self.convert_color_spaces(undistorted)
        otsu, adaptive = self.get_binarization_masks(gray)

        return {
            'bgr': undistorted,
            'gray': gray,
            'hsv': hsv,
            'lab': lab,
            'otsu': otsu,
            'adaptive': adaptive
        }
