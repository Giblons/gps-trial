# Vision-Based Navigation (VBN) System

A modular, multi-threaded Vision-Based Navigation system designed to replace a jammed GPS on a UAV. The system captures live video from a downward-facing MIPI camera, utilizes advanced computer vision to match features against offline raster (MBTiles) and vector (GeoJSON) databases, and computes absolute WGS84 coordinates. Finally, it spoofs a physical GPS module by emitting standard NMEA 0183 sentences over UART to a flight controller.

## Target Hardware
Designed and optimized for the **RADXA Zero 3W** (Rockchip RK3566, ARM Cortex-A53) running Linux, but compatible with other SBCs providing standard MIPI/USB video sources and serial UART capabilities.

## Modular Architecture
The system relies on a clean, modular Python 3 architecture:
- `main.py`: The entrypoint that ties the threads and computer vision logic together.
- `camera_capture.py`: A dedicated thread capturing zero-latency frames from a MIPI camera (or simulated MP4 file).
- `preprocessing.py`: Handles fisheye undistortion using precomputed matrices for ARM NEON SIMD optimization, and yields multi-modal binarization masks (Otsu/Adaptive Gaussian).
- `matching.py`: Executes Raster ORB/FLANN feature matching and contains a fallback system for shape-based semantic matching against Vector geometry.
- `database.py`: Handles Slippy Map tile conversions and querying SQLite MBTiles and GeoJSON maps dynamically through an expanding radius protocol.
- `transformation.py`: Extracts the homography matrix and converts it into standard WGS84 geographic coordinates.
- `nmea_emitter.py`: A dedicated thread buffering and writing `$GPGGA` and `$GPRMC` NMEA serial data to the Pixhawk.

## Prerequisites & Installation

### 1. OS & Packages
```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 2. Python Dependencies
Install the requirements:
```bash
pip3 install numpy opencv-python pyserial
```

## Running the System

Start the autonomous VBN script directly with Python:

```bash
python3 main.py --mbtiles my_satellite_map.mbtiles --geojson my_semantic_map.geojson
```

Upon execution, the multi-threaded system will run endlessly in the background.

## Generating/Acquiring Offline Maps
If you do not have pre-existing maps for your flight area, you can acquire them using the following tools:

- **MBTiles (Raster)**: Use [Mobile Atlas Creator (MOBAC)](http://mobac.sourceforge.net/) or [QGIS](https://qgis.org/) to select your flight region, choose a satellite map source, and export it as an "MBTiles SQLite" file.
- **GeoJSON (Vector)**: Navigate to [Overpass Turbo](https://overpass-turbo.eu/), focus the map on your flight area, and run a query for features like `highway=*` or `building=*`. Export the results directly as GeoJSON.

## Testing & Flight Simulation

### 1. Generating Sample Databases
To test the system without downloading massive mapping files, run this script to generate mock databases:

```bash
python3 download_sample_data.py
```
This will create `sample_satellite.mbtiles` and `sample_map.geojson`.

### 2. Simulating a Flight with MP4 Footage
If you have recorded a previous flight and want to test the algorithm offline using that video instead of a live camera:

```bash
python3 main.py --video my_flight_recording.mp4 --mbtiles sample_satellite.mbtiles --geojson sample_map.geojson
```
