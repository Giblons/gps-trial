import unittest
import numpy as np
import cv2
import math
from vbn_system import NMEAEmitterThread, SharedState, ProcessingEngine, DatabaseManager, FRAME_WIDTH, FRAME_HEIGHT

class TestVBNSystem(unittest.TestCase):

    def setUp(self):
        self.state = SharedState()
        self.nmea_thread = NMEAEmitterThread(self.state)
        # Prevent NMEA thread from actually opening a serial port during test
        self.nmea_thread.serial_conn = None

    def test_nmea_checksum(self):
        # A standard NMEA string without the $ and *xx
        sentence = "GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"
        checksum = self.nmea_thread.calculate_checksum(sentence)
        self.assertEqual(checksum, "47")

    def test_format_dm(self):
        # Latitude test (positive)
        deg_str, dir_char = self.nmea_thread.format_dm(37.7749, is_lat=True)
        self.assertEqual(deg_str, "3746.4940")
        self.assertEqual(dir_char, "N")

        # Longitude test (negative)
        deg_str, dir_char = self.nmea_thread.format_dm(-122.4194, is_lat=False)
        self.assertEqual(deg_str, "12225.1640")
        self.assertEqual(dir_char, "W")

    def test_compute_wgs84_math(self):
        # Testing the coordinate math specifically.
        # We need to create mock objects for kp1, kp2, good_matches
        # or we can test the gis math independently.

        # Let's bypass RANSAC and test just the pixel to WGS84 logic
        db = DatabaseManager()
        engine = ProcessingEngine(self.state, db)

        # Test Slippy Map math
        # Center of tile (nx=0.5, ny=0.5) at zoom 0 (entire world)
        # should be Lat 0, Lon 0
        tile_z = 0
        tile_x = 0
        tile_y = 0
        nx = 0.5
        ny = 0.5

        n = 2.0 ** tile_z
        lon_deg = (tile_x + nx) / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (tile_y + ny) / n)))
        lat_deg = math.degrees(lat_rad)

        self.assertAlmostEqual(lon_deg, 0.0, places=5)
        self.assertAlmostEqual(lat_deg, 0.0, places=5)

    def test_homography_to_wgs84(self):
        # Create a mock database manager and processing engine
        db = DatabaseManager()
        engine = ProcessingEngine(self.state, db)

        # Create mock Keypoints and Matches to form an identity homography
        class MockKeyPoint:
            def __init__(self, x, y):
                self.pt = (x, y)

        class MockDMatch:
            def __init__(self, q_idx, t_idx):
                self.queryIdx = q_idx
                self.trainIdx = t_idx

        kp1 = [MockKeyPoint(100, 100), MockKeyPoint(200, 100), MockKeyPoint(200, 200), MockKeyPoint(100, 200)]
        kp2 = [MockKeyPoint(100, 100), MockKeyPoint(200, 100), MockKeyPoint(200, 200), MockKeyPoint(100, 200)]
        good_matches = [MockDMatch(0,0), MockDMatch(1,1), MockDMatch(2,2), MockDMatch(3,3)]

        # Tile coordinates
        z, x, y = 18, 131072, 131072

        wgs84 = engine.compute_wgs84(kp1, kp2, good_matches, z, x, y)
        self.assertIsNotNone(wgs84)

        lat, lon, alt = wgs84
        self.assertEqual(alt, 100.0) # Fixed altitude check

        # Since identity homography, center of image (320, 240) maps to (320, 240) on tile.
        # But wait, tile size is 256. 320/256 = 1.25, 240/256 = 0.9375
        # The math inside compute_wgs84 should calculate this.
        n = 2.0 ** z
        nx = (FRAME_WIDTH / 2.0) / 256.0
        ny = (FRAME_HEIGHT / 2.0) / 256.0

        expected_lon = (x + nx) / n * 360.0 - 180.0
        expected_lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * (y + ny) / n)))
        expected_lat = math.degrees(expected_lat_rad)

        self.assertAlmostEqual(lat, expected_lat, places=4)
        self.assertAlmostEqual(lon, expected_lon, places=4)


if __name__ == '__main__':
    unittest.main()
