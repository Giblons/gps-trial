# Vision-Based Navigation (VBN) System

An autonomous, multi-threaded Vision-Based Navigation system designed to replace a jammed GPS on a UAV. The system captures live video from a downward-facing MIPI camera, utilizes computer vision to match features against offline raster (MBTiles) and vector (GeoJSON) databases, and computes absolute WGS84 coordinates. Finally, it spoofs a physical GPS module by emitting standard NMEA 0183 sentences over UART to a flight controller.

## Target Hardware
Designed and optimized for the **RADXA Zero 3W** (Rockchip RK3566, ARM Cortex-A53) running Linux, but compatible with other SBCs providing standard MIPI/USB video sources and serial UART capabilities.

## Architecture
The system consists of a multi-threaded Python 3 architecture to ensure zero latency:
1.  **Capture Thread**: Continually grabs frames from the camera, keeping only the most recent frame.
2.  **Processing Engine**: Undistorts the frame, performs global binarization, matches it against offline mapping databases using ORB/FLANN and semantic shape logic, and transforms homography matrices into precise WGS84 slippy map coordinates.
3.  **NMEA Emitter Thread**: Asynchronously formats the coordinates into standard `$GPGGA` and `$GPRMC` NMEA strings and pushes them to the flight controller over UART.

## Prerequisites & Installation

### 1. OS & Packages
Ensure you are running a compatible Linux distribution (e.g., Debian/Ubuntu based).
Update your system packages and install Python 3 and pip.

```bash
sudo apt update
sudo apt install python3 python3-pip
```

### 2. Python Dependencies
The VBN system requires OpenCV (for computer vision and SIMD optimizations), Numpy (for array maths), and PySerial (for UART communication).

Install the requirements:
```bash
pip3 install numpy opencv-python pyserial
```
*Note: Depending on your specific OS build for the Radxa Zero 3W, you may need `opencv-python-headless` instead or prefer to install OpenCV via `apt` (e.g., `sudo apt install python3-opencv`) to guarantee hardware-accelerated GStreamer pipelines.*

## Database Setup (Offline Maps)
This system relies entirely on offline, pre-downloaded map data. Before field operation, populate the local databases.

### 1. Raster Database (MBTiles)
You must provide an MBTiles file (an SQLite database) containing satellite imagery structured in standard web-mercator tile coordinates (Slippy Map).
- You can specify the file path when running the system using the `--mbtiles` argument.

### 2. Vector Database (GeoJSON)
Optionally, provide a GeoJSON file with semantic vector shapes (roads, buildings, rivers) to aid the structural fallback matcher.
- You can specify the file path when running the system using the `--geojson` argument.

## Configuration
Edit the constant configurations at the top of `vbn_system.py` to match your specific hardware parameters:

*   **Camera Configuration**:
    *   `CAMERA_INDEX`: Adjust this if using a GStreamer pipeline string (for hardware ISP access) or standard V4L2 device integer (e.g., `0`).
    *   `K` & `D`: Replace these default arrays with your actual intrinsic camera matrix and distortion coefficients obtained via a standard OpenCV chessboard calibration process.
*   **UART Telemetry Configuration**:
    *   `SERIAL_PORT`: Set this to your active UART device (e.g., `/dev/ttyS0` or `/dev/ttyUSB0`). **Ensure the user running the script has permissions to access this port** (`sudo usermod -a -G dialout $USER`).
    *   `BAUD_RATE`: Set to match your flight controller's GPS port baud rate (typically `115200` or `38400`).

## Running the System

Start the autonomous VBN script directly with Python, passing the paths to your local databases:

```bash
python3 vbn_system.py --mbtiles my_satellite_map.mbtiles --geojson my_semantic_map.geojson
```

Upon execution, the terminal will display startup messages and any critical UART connection warnings. The multi-threaded system will then run endlessly in the background.

## Generating/Acquiring Offline Maps
If you do not have pre-existing maps for your flight area, you can acquire them using the following tools:

- **MBTiles (Raster)**: Use [Mobile Atlas Creator (MOBAC)](http://mobac.sourceforge.net/) or [QGIS](https://qgis.org/) to select your flight region, choose a satellite map source, and export it as an "MBTiles SQLite" file.
- **GeoJSON (Vector)**: Navigate to [Overpass Turbo](https://overpass-turbo.eu/), focus the map on your flight area, and run a query for features like `highway=*` or `building=*`. Export the results directly as GeoJSON.

## Testing & Flight Simulation

### 1. Generating Sample Databases
To immediately test the system without downloading massive mapping files, we provide a script to generate mock databases. Run this first:

```bash
python3 download_sample_data.py
```
This will create `sample_satellite.mbtiles` and `sample_map.geojson` in your directory.

### 2. Simulating a Flight with MP4 Footage
If you have recorded a previous flight and want to test the multi-modal matching algorithm offline using that video instead of a live MIPI camera, use the `simulate_flight.py` script:

```bash
python3 simulate_flight.py --video my_flight_recording.mp4 --mbtiles sample_satellite.mbtiles --geojson sample_map.geojson
```
*Note: Ensure the `--mbtiles` and `--geojson` paths match the geographical area of your video recording for accurate NMEA WGS84 outputs.*

### 3. Unit Tests
To verify the complex math transforms and NMEA string formatting without active camera hardware or databases, run the unit test suite:

```bash
PYTHONPATH=. python3 test_vbn_system.py
```
