import time
import sys
import argparse
import cv2
from camera_capture import CameraCaptureThread
from preprocessing import Preprocessor
from database import DatabaseManager
from matching import Matcher
from transformation import CoordinateTransformer
from nmea_emitter import NMEAEmitterThread

def main():
    """
    Main Processing Engine (Core): Pulls the frame and executes the core computer vision,
    multi-database matching, and geospatial mathematical pipeline.
    Target Hardware: RADXA Zero 3W (Rockchip RK3566, ARM Cortex-A53) running Linux.
    """
    parser = argparse.ArgumentParser(description="Autonomous Vision-Based Navigation (VBN) System")
    parser.add_argument("--mbtiles", default="sample_satellite.mbtiles", help="Path to MBTiles raster database")
    parser.add_argument("--geojson", default="sample_map.geojson", help="Path to GeoJSON vector database")
    parser.add_argument("--port", default="/dev/ttyS0", help="Serial port for NMEA spoofing")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    parser.add_argument("--video", type=str, default=None, help="Path to an optional video file to use instead of live camera")
    args = parser.parse_args()

    print("[Core] Initializing VBN System...")

    # Initialize components
    # Using 640x480 or matching standard video resolution
    FRAME_WIDTH, FRAME_HEIGHT = 640, 480

    preprocessor = Preprocessor(img_shape=(FRAME_WIDTH, FRAME_HEIGHT))
    db_manager = DatabaseManager(mbtiles_path=args.mbtiles, geojson_path=args.geojson)
    matcher = Matcher(min_confidence=0.7)
    transformer = CoordinateTransformer()

    # Initialize multi-threaded architecture
    camera_thread = CameraCaptureThread(camera_index=0, width=FRAME_WIDTH, height=FRAME_HEIGHT, fps=30, video_path=args.video)
    nmea_thread = NMEAEmitterThread(port=args.port, baudrate=args.baud, hz=5)

    # Start threads
    camera_thread.start()
    nmea_thread.start()

    # Initial "last known good" coordinates (San Francisco Example)
    last_lon, last_lat = -122.4194, 37.7749
    zoom_level = 18

    # State tracking for Expanding Radius
    current_search_radius = 0
    max_search_radius = 10

    print("[Core] Entering main processing loop.")
    try:
        while True:
            start_time = time.time()

            # Pull latest frame from mutex-locked buffer
            frame = camera_thread.get_latest_frame()
            if frame is None:
                if not camera_thread.is_alive():
                    print("[Core] Camera thread died. Exiting.")
                    break
                time.sleep(0.01)
                continue

            # Resize if necessary to match pipeline
            if frame.shape[1] != FRAME_WIDTH or frame.shape[0] != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            # 1. Advanced Preprocessing Pipeline (SIMD optimized)
            modalities = preprocessor.preprocess_pipeline(frame)

            # 2. Multi-Tiered Database Integration & Expanding Radius Search
            match_found = False
            best_confidence = 0.0
            best_lon, best_lat = 0.0, 0.0

            # Iterate through the expanding radius tiles
            for xtile, ytile, ref_img in db_manager.expanding_radius_search(last_lon, last_lat, zoom_level, radius=current_search_radius, max_radius=current_search_radius):

                # 3. Multi-Modal Feature Matching & Confidence Scoring
                # Raster Map Match
                H, base_confidence, kp1, kp2, good_matches = matcher.process_raster_match(modalities['gray'], cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY))

                # Vector Shape Semantic Fallback
                # Evaluates structural shapes to artificially boost confidence
                semantic_boost = matcher.process_semantic_vector_match(modalities['adaptive'], db_manager.vector_data)

                confidence = base_confidence + semantic_boost

                if H is not None and confidence >= matcher.min_confidence:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # 4. Coordinate Transformation (Homography to WGS84)
                        tile_lon, tile_lat = db_manager.tile_to_lonlat(xtile, ytile, zoom_level)
                        best_lon, best_lat = transformer.process_transformation(H, (FRAME_WIDTH, FRAME_HEIGHT), tile_lon, tile_lat, zoom_level)
                        match_found = True
                        break

            if match_found:
                print(f"[Core] Fix Acquired: Lat {best_lat:.6f}, Lon {best_lon:.6f} (Conf: {best_confidence:.2f})")
                nmea_thread.update_coordinates(best_lat, best_lon, confidence=best_confidence)
                last_lon, last_lat = best_lon, best_lat
                current_search_radius = 0 # Reset search radius on fix
            else:
                print(f"[Core] Fix Lost. Expanding search radius to {min(current_search_radius + 1, max_search_radius)}...")
                nmea_thread.update_coordinates(last_lat, last_lon, confidence=0.0)
                current_search_radius = min(current_search_radius + 1, max_search_radius)

            # Rate limit main engine loop to prevent CPU pinning
            elapsed = time.time() - start_time
            if elapsed < 0.1:
                time.sleep(0.1 - elapsed)

    except KeyboardInterrupt:
        print("\\n[Core] Shutting down VBN System...")
    finally:
        camera_thread.stop()
        nmea_thread.stop()
        db_manager.close()
        sys.exit(0)

if __name__ == "__main__":
    main()
