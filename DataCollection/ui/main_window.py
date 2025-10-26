##
# @file main_window.py
# @brief Main window for the data collection PyQt5 application.
from PyQt5.QtWidgets import QMainWindow, QPushButton, QWidget, QVBoxLayout, QMessageBox
import pyqtgraph as pg
import numpy as np
import serial
import threading

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 152000
WINDOW_SIZE = 5000  # Show last 5 seconds at 1000 Hz

##
# @class MainWindow
# @brief Main window class for data collection and live plotting.
class MainWindow(QMainWindow):
    ##
    # @brief Constructor for MainWindow.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Collection")

        central_widget = QWidget()
        layout = QVBoxLayout()

        self.collect_button = QPushButton("Collect Data")
        self.collect_button.clicked.connect(self.collect_data)
        layout.addWidget(self.collect_button)

        # PyQtGraph plot widgets for each signal
        self.plot_eeg_adc = pg.PlotWidget(title="EEG (ADC)")
        self.plot_eeg_v = pg.PlotWidget(title="EEG (V)")
        self.plot_ecg_adc = pg.PlotWidget(title="ECG (ADC)")
        self.plot_ecg_v = pg.PlotWidget(title="ECG (V)")
        self.plot_spo2 = pg.PlotWidget(title="SpO2 (%)")

        layout.addWidget(self.plot_eeg_adc)
        layout.addWidget(self.plot_eeg_v)
        layout.addWidget(self.plot_ecg_adc)
        layout.addWidget(self.plot_ecg_v)
        layout.addWidget(self.plot_spo2)

        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Data buffers
        self.eeg_adc = np.zeros(WINDOW_SIZE)
        self.eeg_v = np.zeros(WINDOW_SIZE)
        self.ecg_adc = np.zeros(WINDOW_SIZE)
        self.ecg_v = np.zeros(WINDOW_SIZE)
        self.spo2_percent = np.zeros(WINDOW_SIZE)

        # Plot curves
        self.curve_eeg_adc = self.plot_eeg_adc.plot(pen='b')
        self.curve_eeg_v = self.plot_eeg_v.plot(pen='c')
        self.curve_ecg_adc = self.plot_ecg_adc.plot(pen='g')
        self.curve_ecg_v = self.plot_ecg_v.plot(pen='m')
        self.curve_spo2 = self.plot_spo2.plot(pen='r')

        # Timer for UI updates
        self.timer = pg.QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)

        # Threading
        self.serial_thread = None
        self.stop_event = threading.Event()

    ##
    # @brief Starts data collection and serial reading thread.
    def collect_data(self):
        self.stop_event.clear()
        self.serial_thread = threading.Thread(target=self.read_serial)
        self.serial_thread.start()
        self.timer.start(50)
        QMessageBox.information(self, "Data Collection", "Data collection started!")

    ##
    # @brief Reads serial data and updates buffers.
    def read_serial(self):
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            while not self.stop_event.is_set():
                line = ser.readline().decode('utf-8', errors='replace').strip()
                # Expected format: t,eeg_adc,ecg_adc,spo2_percent
                parts = line.split(',')
                if len(parts) >= 4:
                    try:
                        t = int(parts[0])
                        eeg_adc_val = int(parts[1])
                        ecg_adc_val = int(parts[2])
                        spo2_val = float(parts[3])
                        # Convert ADC to voltage
                        eeg_v_val = (eeg_adc_val / 4095) * 3.3
                        ecg_v_val = ((ecg_adc_val - 2024) / 4095) * 3.3
                    except Exception:
                        eeg_adc_val = ecg_adc_val = 0
                        spo2_val = eeg_v_val = ecg_v_val = 0
                    self.eeg_adc[:-1] = self.eeg_adc[1:]
                    self.eeg_adc[-1] = eeg_adc_val
                    self.eeg_v[:-1] = self.eeg_v[1:]
                    self.eeg_v[-1] = eeg_v_val
                    self.ecg_adc[:-1] = self.ecg_adc[1:]
                    self.ecg_adc[-1] = ecg_adc_val
                    self.ecg_v[:-1] = self.ecg_v[1:]
                    self.ecg_v[-1] = ecg_v_val
                    self.spo2_percent[:-1] = self.spo2_percent[1:]
                    self.spo2_percent[-1] = spo2_val
            ser.close()
        except Exception as e:
            print(f"Serial error: {e}")

    ##
    # @brief Updates the plot widgets with new data.
    def update_plot(self):
        self.curve_eeg_adc.setData(self.eeg_adc)
        self.curve_eeg_v.setData(self.eeg_v)
        self.curve_ecg_adc.setData(self.ecg_adc)
        self.curve_ecg_v.setData(self.ecg_v)
        self.curve_spo2.setData(self.spo2_percent)

    ##
    # @brief Handles window close event and stops threads.
    # @param event The close event.
    def closeEvent(self, event):
        self.stop_event.set()
        if self.serial_thread is not None:
            self.serial_thread.join()
        event.accept()
