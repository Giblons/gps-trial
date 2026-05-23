import urllib.request
import json
import sqlite3
import os
import cv2
import numpy as np

def download_osm_geojson(output_file="sample_map.geojson"):
    print("[Downloader] Fetching sample Vector database (GeoJSON) from OpenStreetMap Overpass API...")
    # A small bounding box for a sample area
    overpass_url = "http://overpass-api.de/api/interpreter"
    overpass_query = """
    [out:json];
    (
      way["highway"](37.7749,-122.4194,37.7849,-122.4094);
      way["building"](37.7749,-122.4194,37.7849,-122.4094);
    );
    out body;
    >;
    out skel qt;
    """

    try:
        req = urllib.request.Request(overpass_url, data=overpass_query.encode('utf-8'), headers={'User-Agent': 'VBN-Testing-Script'})
        response = urllib.request.urlopen(req)
        data = json.loads(response.read().decode('utf-8'))

        # Convert very basic OSM JSON to a mock GeoJSON structure for our test
        geojson = {
            "type": "FeatureCollection",
            "features": []
        }
        for element in data.get('elements', []):
            if element['type'] == 'way':
                geojson["features"].append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0,0], [1,1], [0,1]]] # Mock coordinates for simplicity of the structural test
                    }
                })

        with open(output_file, 'w') as f:
            json.dump(geojson, f, indent=2)
        print(f"[Downloader] Saved sample GeoJSON to {output_file}")
    except Exception as e:
        print(f"[Downloader] Failed to fetch GeoJSON: {e}")

def create_mock_mbtiles(output_file="sample_satellite.mbtiles"):
    print("[Downloader] Generating sample Raster database (MBTiles SQLite)...")
    if os.path.exists(output_file):
        os.remove(output_file)

    conn = sqlite3.connect(output_file)
    cursor = conn.cursor()

    # MBTiles standard schema
    cursor.execute("CREATE TABLE metadata (name text, value text);")
    cursor.execute("CREATE TABLE tiles (zoom_level integer, tile_column integer, tile_row integer, tile_data blob);")

    cursor.execute("INSERT INTO metadata VALUES ('name', 'Sample Satellite Database');")
    cursor.execute("INSERT INTO metadata VALUES ('format', 'jpg');")

    # Generate a random noise tile to act as a mock satellite image
    mock_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', mock_img)
    blob_data = buffer.tobytes()

    # Insert at our assumed initial search tile in vbn_system.py (z=18, x=131072, y=131072)
    # Note: TMS flips the Y axis. MBTiles requires TMS.
    zoom = 18
    tile_col = 131072
    tile_row_xyz = 131072
    tms_row = (2 ** zoom) - 1 - tile_row_xyz

    cursor.execute("INSERT INTO tiles VALUES (?, ?, ?, ?)", (zoom, tile_col, tms_row, blob_data))

    conn.commit()
    conn.close()
    print(f"[Downloader] Saved mock MBTiles to {output_file}")

if __name__ == "__main__":
    print("--- VBN System Sample Data Provisioner ---")
    download_osm_geojson()
    create_mock_mbtiles()
    print("--- Provisioning Complete ---")
