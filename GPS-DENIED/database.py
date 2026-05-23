import sqlite3
import json
import cv2
import numpy as np
import math

class DatabaseManager:
    """
    Multi-Tiered Database Integration & Expanding Search Radius
    Handles SQLite/MBTiles raster data and GeoJSON vector data.
    """
    def __init__(self, mbtiles_path="map.mbtiles", geojson_path="map.geojson"):
        self.mbtiles_path = mbtiles_path
        self.geojson_path = geojson_path
        self.conn = None
        self.vector_data = None

        try:
            self.conn = sqlite3.connect(self.mbtiles_path, check_same_thread=False)
            print(f"[DB] Connected to MBTiles: {self.mbtiles_path}")
        except sqlite3.Error as e:
            print(f"[DB] Error connecting to MBTiles {self.mbtiles_path}: {e}")

        try:
            with open(self.geojson_path, 'r') as f:
                self.vector_data = json.load(f)
            print(f"[DB] Loaded Vector Database: {self.geojson_path}")
        except Exception as e:
            print(f"[DB] Warning: Could not load GeoJSON vector DB {self.geojson_path}: {e}")

    def lonlat_to_tile(self, lon, lat, zoom):
        """Standard Slippy Map math to convert coordinates to tile indices."""
        n = 2.0 ** zoom
        xtile = int((lon + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n)
        return xtile, ytile

    def tile_to_lonlat(self, xtile, ytile, zoom):
        """Returns the NW corner of the tile."""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return lon_deg, lat_deg

    def query_tile(self, xtile, ytile, zoom):
        """Retrieves a specific tile from the MBTiles SQLite DB."""
        if self.conn is None:
            return None
        # MBTiles uses TMS, so we invert the Y axis
        tms_y = (2 ** zoom) - 1 - ytile
        cursor = self.conn.cursor()
        try:
            cursor.execute(
                "SELECT tile_data FROM tiles WHERE zoom_level=? AND tile_column=? AND tile_row=?",
                (zoom, xtile, tms_y)
            )
            row = cursor.fetchone()
            if row and row[0]:
                nparr = np.frombuffer(row[0], np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                return img
        except sqlite3.Error as e:
            print(f"[DB] Query Error: {e}")
        return None

    def expanding_radius_search(self, lon, lat, zoom, radius=1, max_radius=3):
        """
        Expanding Radius Search Protocol.
        Starts querying local tiles, expanding outward if confidence is low.
        Yields (xtile, ytile, image) tuples.
        """
        center_x, center_y = self.lonlat_to_tile(lon, lat, zoom)

        current_radius = radius
        while current_radius <= max_radius:
            print(f"[DB] Searching radius {current_radius} around tile {center_x},{center_y}")
            for dx in range(-current_radius, current_radius + 1):
                for dy in range(-current_radius, current_radius + 1):
                    x = center_x + dx
                    y = center_y + dy
                    img = self.query_tile(x, y, zoom)
                    if img is not None:
                        yield x, y, img
            current_radius += 1

    def close(self):
        if self.conn:
            self.conn.close()
