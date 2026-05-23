import serial
import threading
import time
import datetime
import math

class NMEAEmitterThread(threading.Thread):
    """
    NMEA Emitter Thread: Asynchronously formats derived WGS84 coordinates into valid
    NMEA strings and streams them to the Pixhawk via UART (e.g., /dev/ttyS0).
    """
    def __init__(self, port='/dev/ttyS0', baudrate=115200, hz=5):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.hz = hz
        self.period = 1.0 / self.hz

        self.lock = threading.Lock()
        self.current_lat = 0.0
        self.current_lon = 0.0
        self.current_alt = 0.0
        self.has_fix = False
        self.satellites = 0

        self.running = False
        self.daemon = True

        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=1)
            print(f"[NMEAEmitter] Connected to {self.port} at {self.baudrate} baud.")
        except serial.SerialException as e:
            print(f"[NMEAEmitter] Warning: Could not open serial port {self.port}: {e}")
            self.serial_conn = None

    def update_coordinates(self, lat, lon, alt=0.0, confidence=1.0):
        """Updates the coordinates to be emitted. Called by the core engine."""
        with self.lock:
            self.current_lat = lat
            self.current_lon = lon
            self.current_alt = alt
            self.has_fix = confidence >= 0.7
            # Spoof satellites based on fix
            self.satellites = 10 if self.has_fix else 0

    def calculate_checksum(self, sentence):
        """Calculates the NMEA checksum."""
        calc_cksum = 0
        for char in sentence:
            calc_cksum ^= ord(char)
        return f"{calc_cksum:02X}"

    def format_lat_lon(self, decimal_degrees, is_lat):
        """Converts decimal degrees to NMEA format (DDMM.MMMM)."""
        if is_lat:
            direction = 'N' if decimal_degrees >= 0 else 'S'
        else:
            direction = 'E' if decimal_degrees >= 0 else 'W'

        abs_deg = abs(decimal_degrees)
        degrees = int(abs_deg)
        minutes = (abs_deg - degrees) * 60

        if is_lat:
            formatted = f"{degrees:02d}{minutes:07.4f}"
        else:
            formatted = f"{degrees:03d}{minutes:07.4f}"

        return formatted, direction

    def generate_gpgga(self, utc_time, lat_str, lat_dir, lon_str, lon_dir):
        """Generates a valid $GPGGA sentence."""
        fix_quality = 1 if self.has_fix else 0
        hdop = "1.0" if self.has_fix else "99.99"

        # Format: $GPGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx*hh
        sentence = f"GPGGA,{utc_time},{lat_str},{lat_dir},{lon_str},{lon_dir},{fix_quality},{self.satellites:02d},{hdop},{self.current_alt:.1f},M,0.0,M,,"
        checksum = self.calculate_checksum(sentence)
        return f"${sentence}*{checksum}\r\n"

    def generate_gprmc(self, utc_time, date_str, lat_str, lat_dir, lon_str, lon_dir):
        """Generates a valid $GPRMC sentence."""
        status = 'A' if self.has_fix else 'V'
        speed_knots = "0.0" # Could be calculated from history if needed
        course = "0.0"

        # Format: $GPRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,x.x,a*hh
        sentence = f"GPRMC,{utc_time},{status},{lat_str},{lat_dir},{lon_str},{lon_dir},{speed_knots},{course},{date_str},,,"
        checksum = self.calculate_checksum(sentence)
        return f"${sentence}*{checksum}\r\n"

    def run(self):
        self.running = True
        print("[NMEAEmitter] Emitter thread started.")
        while self.running:
            start_time = time.time()

            with self.lock:
                lat = self.current_lat
                lon = self.current_lon
                has_fix = self.has_fix

            if has_fix:
                now = datetime.datetime.utcnow()
                utc_time = now.strftime("%H%M%S.%f")[:-4]
                date_str = now.strftime("%d%m%y")

                lat_str, lat_dir = self.format_lat_lon(lat, True)
                lon_str, lon_dir = self.format_lat_lon(lon, False)

                gpgga = self.generate_gpgga(utc_time, lat_str, lat_dir, lon_str, lon_dir)
                gprmc = self.generate_gprmc(utc_time, date_str, lat_str, lat_dir, lon_str, lon_dir)

                if self.serial_conn and self.serial_conn.is_open:
                    try:
                        self.serial_conn.write(gpgga.encode('ascii'))
                        self.serial_conn.write(gprmc.encode('ascii'))
                    except Exception as e:
                        print(f"[NMEAEmitter] Serial write error: {e}")
                else:
                    # For debugging without hardware
                    # print(f"[NMEA Spoofer] {gpgga.strip()} | {gprmc.strip()}")
                    pass

            # Sleep to maintain desired Hz
            elapsed = time.time() - start_time
            sleep_time = self.period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        """Stops the emitter thread and closes the serial port."""
        self.running = False
        self.join()
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        print("[NMEAEmitter] Emitter thread stopped.")
