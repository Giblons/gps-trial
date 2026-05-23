import cv2
import threading
import time

class CameraCaptureThread(threading.Thread):
    """
    Capture Thread: Reads a continuous video stream from the MIPI camera
    using cv2.VideoCapture. It places the latest frame into a thread-safe,
    mutex-locked buffer, actively discarding stale frames to ensure zero-latency processing.
    Targeting RADXA Zero 3W (Rockchip RK3566, ARM Cortex-A53).
    """
    def __init__(self, camera_index=0, width=1920, height=1080, fps=30, video_path=None):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.video_path = video_path
        self.is_video_file = False

        if self.video_path is not None:
            print(f"[CameraCapture] Opening video file: {self.video_path}")
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_video_file = True
            if not self.cap.isOpened():
                print(f"[CameraCapture] Error opening video file {self.video_path}")
        else:
            # GStreamer pipeline string for MIPI camera on Rockchip
            # This pipeline is a common example for V4L2 on such ARM boards.
            self.pipeline = (
                f"v4l2src device=/dev/video{self.camera_index} ! "
                f"video/x-raw, format=NV12, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
                "videoconvert ! appsink drop=true max-buffers=1"
            )

            # Open video capture
            # Fallback to standard V4L2 if GStreamer fails or is unavailable
            self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                print(f"[CameraCapture] GStreamer pipeline failed, falling back to V4L2 index {self.camera_index}")
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.daemon = True # Thread dies when main thread exits

    def run(self):
        self.running = True
        print("[CameraCapture] Capture thread started.")

        # Calculate delay needed to simulate real-time framerate for video files
        frame_delay = 1.0 / self.fps if self.is_video_file else 0

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret and self.is_video_file:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            if ret and frame is not None:
                with self.lock:
                    # Update the latest frame, discarding the old one
                    self.latest_frame = frame
            else:
                # If frame drop or error, sleep slightly to yield CPU
                time.sleep(0.01)
                continue

            # Simulate real-time playback for video files
            if self.is_video_file:
                elapsed = time.time() - start_time
                sleep_time = frame_delay - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    def get_latest_frame(self):
        """
        Pulls the latest frame from the buffer.
        Returns None if no frame is available.
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """Stops the capture thread and releases resources."""
        self.running = False
        self.join()
        if self.cap.isOpened():
            self.cap.release()
        print("[CameraCapture] Capture thread stopped.")
