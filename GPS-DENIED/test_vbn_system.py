import unittest
import numpy as np
import cv2
import math
from nmea_emitter import NMEAEmitterThread
from database import DatabaseManager
from transformation import CoordinateTransformer

class TestVBNSystem(unittest.TestCase):

    def setUp(self):
        self.nmea_thread = NMEAEmitterThread(port="/dev/ttyS0", baudrate=115200, hz=5)
        # Prevent NMEA thread from actually opening a serial port during test
        self.nmea_thread.serial_conn = None

    def test_nmea_checksum(self):
        # A standard NMEA string without the $ and *xx
        sentence = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
        checksum = self.nmea_thread.calculate_checksum(sentence)
        self.assertEqual(checksum, "47")

    def test_format_lat_lon(self):
        # Latitude test (positive)
        deg_str, dir_char = self.nmea_thread.format_lat_lon(37.7749, is_lat=True)
        self.assertEqual(deg_str, "3746.4940")
        self.assertEqual(dir_char, "N")

        # Longitude test (negative)
        deg_str, dir_char = self.nmea_thread.format_lat_lon(-122.4194, is_lat=False)
        self.assertEqual(deg_str, "12225.1640")
        self.assertEqual(dir_char, "W")

    def test_compute_wgs84_math(self):
        # Let's test the Slippy map math directly from the database manager
        db = DatabaseManager(mbtiles_path="mock.mbtiles", geojson_path="mock.geojson")

        # Center of tile (nx=0.5, ny=0.5) at zoom 0 (entire world)
        # Should correspond to near 0,0 but let's test specific tile corner logic
        zoom = 0
        xtile, ytile = 0, 0
        lon, lat = db.tile_to_lonlat(xtile, ytile, zoom)

        self.assertAlmostEqual(lon, -180.0, places=5)
        self.assertAlmostEqual(lat, 85.0511287798066, places=5)

    def test_homography_to_wgs84(self):
        transformer = CoordinateTransformer()

        # Identity homography maps exactly to itself
        H = np.eye(3, dtype=np.float32)

        frame_shape = (640, 480) # W, H
        tile_lon = -122.4194
        tile_lat = 37.7749
        zoom = 18

        wgs84 = transformer.process_transformation(H, frame_shape, tile_lon, tile_lat, zoom)
        self.assertIsNotNone(wgs84)

        lon, lat = wgs84

        # The center of a 640x480 frame is (320, 240)
        # Inside process_transformation, it computes the slippy translation
        TILE_SIZE = 256.0
        n = 2.0 ** zoom

        # Original tile coords
        x_tile = (tile_lon + 180.0) / 360.0 * n
        y_tile = (1.0 - math.asinh(math.tan(math.radians(tile_lat))) / math.pi) / 2.0 * n

        # Projected Center
        px = 320.0
        py = 240.0

        expected_lon = (x_tile + px/TILE_SIZE) / n * 360.0 - 180.0
        expected_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y_tile + py/TILE_SIZE) / n)))
        expected_lat = math.degrees(expected_lat_rad)

        self.assertAlmostEqual(lat, expected_lat, places=2)
        self.assertAlmostEqual(lon, expected_lon, places=2)

if __name__ == '__main__':
    unittest.main()
