import numpy as np
import cv2

class CoordinateTransformer:
    """
    Coordinate Transformation (Homography to WGS84)
    Transforms the center pixel of the live camera ROI into the coordinate space
    of the matched reference tile using the Homography matrix, and then applies
    an affine transformation to translate the pixel into absolute WGS84 coordinates.
    """
    def __init__(self, tile_size=256):
        self.tile_size = tile_size

    def transform_center_pixel(self, H, img_shape):
        """
        Transforms the center pixel (cx, cy) of the live camera ROI
        into the coordinate space of the matched reference tile.
        """
        w, h = img_shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Create homogeneous coordinate for the center pixel
        center_pt = np.array([[[cx, cy]]], dtype=np.float32)

        # Apply perspective transform
        transformed_pt = cv2.perspectiveTransform(center_pt, H)

        px, py = transformed_pt[0][0]
        return px, py

    def pixel_to_wgs84(self, px, py, tile_lon, tile_lat, zoom):
        """
        Applies a GIS standard affine transformation to translate the localized
        pixel into absolute WGS84 coordinates.
        Assume Slippy map tiles have zero rotation/skew.
        mx = C + (px * A) + (py * B) -> Simplified for Web Mercator / Slippy Map
        """
        import math

        # n is the number of tiles across the map at this zoom level
        n = 2.0 ** zoom

        # First, find the exact global tile coordinates (fractional)
        # We start with the base tile integer coordinates
        # To get the integer coordinates, we need to convert the top-left lon/lat back to tile ints
        xtile = int((tile_lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(tile_lat))) / math.pi) / 2.0 * n)

        # Add the fractional pixel offset
        # Note: px and py are relative to the top-left of the matched tile
        global_x = xtile + (px / self.tile_size)
        global_y = ytile + (py / self.tile_size)

        # Convert global fractional tile coordinates back to WGS84 Lon/Lat
        lon_deg = global_x / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * global_y / n)))
        lat_deg = math.degrees(lat_rad)

        return lon_deg, lat_deg

    def process_transformation(self, H, img_shape, tile_lon, tile_lat, zoom):
        """
        Full transformation pipeline.
        Returns (lon, lat) WGS84 coordinates.
        """
        px, py = self.transform_center_pixel(H, img_shape)
        lon, lat = self.pixel_to_wgs84(px, py, tile_lon, tile_lat, zoom)
        return lon, lat
