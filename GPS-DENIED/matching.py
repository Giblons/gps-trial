import cv2
import numpy as np

class Matcher:
    """
    Multi-Modal Feature Matching & Confidence Scoring
    Implements ORB with FLANN (LSH) for raster database matching,
    with a shape-matching fallback for semantic Vector (GeoJSON) geometry.
    """
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence
        self.orb = cv2.ORB_create(nfeatures=2000)

        # Configure FLANN matcher with LSH for fast binary descriptor matching
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features(self, image):
        """Extracts ORB keypoints and descriptors."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, kp1, des1, kp2, des2):
        """Matches features using FLANN and applies Lowe's Ratio Test."""
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []
        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except Exception:
            return []

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Strict Lowe's Ratio Test (ratio < 0.7)
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches

    def calculate_homography_and_confidence(self, kp1, kp2, good_matches):
        """Calculates Planar Perspective Transformation using RANSAC and yields confidence."""
        if len(good_matches) < 4:
            return None, 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None or mask is None:
            return None, 0.0

        inliers = np.sum(mask)
        total_matches = len(good_matches)

        # Score confidence based on pure inlier count (requires 25+ for perfect 1.0 score)
        # Scaled by ratio of inliers.
        ratio = inliers / total_matches if total_matches > 0 else 0
        confidence = min(inliers / 25.0, 1.0) * ratio

        return M, confidence

    def process_raster_match(self, live_gray, reference_gray):
        """Matches a live frame against an MBTiles reference image."""
        kp1, des1 = self.extract_features(live_gray)
        kp2, des2 = self.extract_features(reference_gray)

        good_matches = self.match_features(kp1, des1, kp2, des2)
        H, confidence = self.calculate_homography_and_confidence(kp1, kp2, good_matches)

        return H, confidence, kp1, kp2, good_matches

    def process_semantic_vector_match(self, adaptive_binary, vector_data):
        """
        Semantic Fallback:
        Compares contours from the adaptive Gaussian thresholded frame against
        polygons parsed from the GeoJSON vector database to boost confidence
        in degraded visual environments if structures physically match.
        """
        confidence_boost = 0.0

        if vector_data is None or "features" not in vector_data:
            return confidence_boost

        # Extract shapes from the preprocessed binary frame
        contours, _ = cv2.findContours(adaptive_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return confidence_boost

        # Parse Vector Data into pseudo-contours for matching
        vector_contours = []
        for feature in vector_data.get("features", []):
            geom = feature.get("geometry", {})
            if geom.get("type") in ["Polygon", "MultiPolygon"]:
                coords = geom.get("coordinates", [])
                if coords:
                    for poly in coords:
                        # Simplify geometry to just a rough matching contour
                        pts = np.array(poly, dtype=np.float32)
                        # We mock the spatial projection for performance, focusing on shape structure
                        pts = pts.reshape((-1, 1, 2))
                        vector_contours.append(pts)

        if not vector_contours:
            return confidence_boost

        # Mathematically compare the structural shapes using cv2.matchShapes
        # We look for at least one highly similar structural shape (e.g. building footprint)
        for live_cnt in contours:
            if cv2.contourArea(live_cnt) < 500:
                continue

            for vec_cnt in vector_contours:
                try:
                    # Calculates Hu Moments distance (lower is better, 0 is exact match)
                    similarity = cv2.matchShapes(live_cnt, vec_cnt, cv2.CONTOURS_MATCH_I1, 0.0)

                    if similarity < 0.2: # Very similar shape
                        confidence_boost = max(confidence_boost, 0.2)
                    elif similarity < 0.5: # Somewhat similar shape
                        confidence_boost = max(confidence_boost, 0.1)
                except Exception:
                    pass

        return confidence_boost
