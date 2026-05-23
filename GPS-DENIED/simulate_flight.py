import cv2
import threading
import time
import argparse
from vbn_system import SharedState, ProcessingEngine, NMEAEmitterThread, DatabaseManager, FRAME_WIDTH, FRAME_HEIGHT

# ==============================================================================
# Custom Simulation Capture Thread
# ==============================================================================
class SimulationCaptureThread(threading.Thread):
    """
    Reads continuous video stream from a recorded MP4/MKV video file instead of
    a live MIPI camera for offline flight testing.
    """
    def __init__(self, state: SharedState, video_path: str):
        super().__init__()
        self.state = state
        self.daemon = True
        self.running = True
        self.video_path = video_path
        self.cap = cv2.VideoCapture(self.video_path)

        if not self.cap.isOpened():
            print(f"[Sim] Error: Could not open video file {self.video_path}")
            self.running = False

    def run(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        delay = 1.0 / fps if fps > 0 else 0.033 # Fallback to ~30 FPS

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()

            if ret:
                # Resize to match expected system resolution if necessary
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                with self.state.frame_mutex:
                    self.state.latest_frame = frame
            else:
                # Loop video or stop
                print("[Sim] End of video reached.")
                self.running = False
                break

            # Sleep to match video framerate to avoid processing too fast
            elapsed = time.time() - start_time
            sleep_time = max(0, delay - elapsed)
            time.sleep(sleep_time)

    def stop(self):
        self.running = False
        if hasattr(self, 'cap'):
            self.cap.release()

# ==============================================================================
# Simulation Entrypoint
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VBN Flight Simulation Utility")
    parser.add_argument("--video", type=str, required=True, help="Path to the recorded flight video file (.mp4, .mkv)")
    parser.add_argument("--mbtiles", type=str, default="sample_satellite.mbtiles", help="Path to the MBTiles raster database")
    parser.add_argument("--geojson", type=str, default="sample_map.geojson", help="Path to the GeoJSON vector database")
    args = parser.parse_args()

    print("[VBN] Starting Offline Flight Simulation...")
    print(f"[VBN] Using Video footage: {args.video}")
    print(f"[VBN] Using MBTiles database: {args.mbtiles}")
    print(f"[VBN] Using GeoJSON database: {args.geojson}")

    shared_state = SharedState()
    db_manager = DatabaseManager(mbtiles_path=args.mbtiles, geojson_path=args.geojson)

    sim_thread = SimulationCaptureThread(shared_state, args.video)
    core_thread = ProcessingEngine(shared_state, db_manager)
    nmea_thread = NMEAEmitterThread(shared_state)

    if not sim_thread.running:
        exit(1)

    sim_thread.start()
    core_thread.start()
    nmea_thread.start()

    try:
        while sim_thread.is_alive():
            time.sleep(1)
            # Optional: Display the frame being processed for visual debugging
            with shared_state.frame_mutex:
                frame = shared_state.latest_frame
            if frame is not None:
                cv2.imshow("VBN Flight Simulator", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[VBN] Shutting down simulation...")
        sim_thread.stop()
        core_thread.stop()
        nmea_thread.stop()
        cv2.destroyAllWindows()
