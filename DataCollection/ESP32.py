##
# @file ESP32.py
# @brief Serial data acquisition and plotting for ESP32-based device.
import serial
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import csv
import threading
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 152000
WINDOW_SIZE = 5000  # Show last 5 seconds at 1000 Hz
ADC_MAX = 4095
VREF = 3.3  # Reference voltage for ESP32 ADC

RECORD_DELAY = 10      # seconds before starting recording
RECORD_DURATION = 10800/3  # seconds to record (5 minutes)
CSV_FILENAME = "esp32_recorded_data.csv"
print(f"Saving ESP32 data to: {CSV_FILENAME}")

##
# @brief Converts ADC value to voltage.
# @param adc_val ADC value.
# @return Voltage value.
def adc_to_voltage(adc_val):
    return (adc_val / ADC_MAX) * VREF

##
# @brief Parses a line of serial data.
# @param line String line from serial.
# @return Tuple (timestamp, eeg_adc, ecg_adc, spo2_percent) or None.
def parse_line(line):
    try:
        parts = line.split(',')
        if len(parts) >= 4:
            t, e, c, s = parts[:4]
            return (int(t), int(e), int(c), float(s))
        else:
            return None
    except Exception:
        return None

##
# @brief Reads serial data and updates buffers, optionally writes to CSV.
# @param ser Serial object.
# @param eeg_adc EEG ADC buffer.
# @param eeg_v EEG voltage buffer.
# @param ecg_adc ECG ADC buffer.
# @param ecg_v ECG voltage buffer.
# @param spo2_percent SpO₂ buffer.
# @param stop_event Threading event to stop reading.
# @param csv_writer Optional CSV writer object.
def serial_reader(ser, eeg_adc, eeg_v, ecg_adc, ecg_v, spo2_percent, stop_event, csv_writer=None):
    while not stop_event.is_set():
        line = ser.readline().decode('utf-8', errors='replace').strip()
        data = parse_line(line)
        if data:
            t, e, c, s = data
            eeg_adc[:-1] = eeg_adc[1:]
            eeg_adc[-1] = e
            eeg_v[:-1] = eeg_v[1:]
            eeg_v[-1] = adc_to_voltage(e)
            ecg_adc[:-1] = ecg_adc[1:]
            ecg_adc[-1] = c
            ecg_v[:-1] = ecg_v[1:]
            ecg_v[-1] = adc_to_voltage(c - 2024)
            spo2_percent[:-1] = spo2_percent[1:]
            spo2_percent[-1] = s

            # Write to CSV immediately
            if csv_writer:
                csv_writer.writerow([t, e, c, s])

##
# @brief Runs live plot animation and starts serial reading thread.
# @param ser Serial object.
# @param fig Matplotlib figure.
# @param axs List of axes.
# @param lines List of line objects.
# @param x X-axis data.
# @param canvas FigureCanvas object.
# @return Tuple (animation object, stop_event).
def run_plot(ser, fig, axs, lines, x, canvas):
    eeg_adc = np.zeros(WINDOW_SIZE)
    eeg_v = np.zeros(WINDOW_SIZE)
    ecg_adc = np.zeros(WINDOW_SIZE)
    ecg_v = np.zeros(WINDOW_SIZE)
    spo2_percent = np.zeros(WINDOW_SIZE)
    stop_event = threading.Event()

    # Open CSV file in append mode
    csv_file = open(CSV_FILENAME, 'a', newline='')
    csv_writer = csv.writer(csv_file)
    # Optionally write header if file is empty
    if csv_file.tell() == 0:
        csv_writer.writerow(['timestamp', 'eeg_adc', 'ecg_adc', 'spo2_percent'])

    thread = threading.Thread(target=serial_reader, args=(ser, eeg_adc, eeg_v, ecg_adc, ecg_v, spo2_percent, stop_event, csv_writer))
    thread.daemon = True
    thread.start()

    def update_plot(frame):
        lines[0].set_data(x, eeg_adc)
        lines[1].set_data(x, eeg_v)
        lines[2].set_data(x, ecg_adc)
        lines[3].set_data(x, ecg_v)
        lines[4].set_data(x, spo2_percent)
        for ax in axs:
            ax.figure.canvas.draw()
        if canvas:
            canvas.draw()
        return lines

    ani = animation.FuncAnimation(
        fig, update_plot,
        interval=50, blit=False,
        cache_frame_data=False
    )
    print("FuncAnimation created")  # Debug: animation object created
    fig.tight_layout()
    # Do not call plt.show() here; the canvas is embedded in the PyQt window

    # Ensure CSV file is closed when stop_event is set
    def close_csv_file():
        stop_event.wait()
        csv_file.close()
    threading.Thread(target=close_csv_file, daemon=True).start()

    return ani, stop_event

##
# @brief Opens serial port and starts data collection and plotting.
# @param canvas FigureCanvas object.
# @param fig Matplotlib figure.
# @param axs List of axes.
# @param lines List of line objects.
# @param x X-axis data.
# @return Tuple ((animation, stop_event), serial object) or (None, None), None on failure.
def collect_data(canvas, fig, axs, lines, x):
    print("collect_data called")  # Debug: function entry
    print(f"Attempting to open serial port: {SERIAL_PORT} at {BAUD_RATE}")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("Serial port opened successfully")
        return run_plot(ser, fig, axs, lines, x, canvas), ser
    except (serial.SerialException, OSError) as e:
        print(f"[DEBUG] Connection lost or failed: {e}")
        return (None, None), None
