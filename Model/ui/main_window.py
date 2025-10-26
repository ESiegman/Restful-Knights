##
# @file main_window.py
# @brief Main window for the sleep analysis PyQt5 application.
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QHeaderView, QTableWidgetItem, QTableWidget
from PyQt5.QtCore import Qt
from llama.llama_client import get_response

##
# @class MainWindow
# @brief Main window class for sleep analysis application.
class MainWindow(QMainWindow):
    ##
    # @brief Constructor for MainWindow.
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My PyQt5 Application")
        self.setGeometry(250, 250, 600, 400)
        self.init_ui()

    ##
    # @brief Initializes the UI layout and widgets.
    def init_ui(self):
        self.center()

    ##
    # @brief Centers the window on the screen and sets up widgets/layouts.
    def center(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Our name goes here")
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 5px;")
        main_layout.addWidget(title_label, alignment=Qt.AlignHCenter | Qt.AlignTop)

        self.llm_output = QTextEdit()
        self.llm_output.setPlainText("Lets analyze your sleep data!")
        self.llm_output.setReadOnly(True)
        self.llm_output.setStyleSheet("font-size: 14px;")
        main_layout.addWidget(self.llm_output)

        results_label = QLabel("Results")
        results_label.setAlignment(Qt.AlignHCenter)
        results_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(results_label, alignment=Qt.AlignHCenter)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(2)
        self.results_table.setHorizontalHeaderLabels(["Time", "event"])
        self.results_table.setMaximumHeight(150)
        self.results_table.setEditTriggers(QTableWidget.NoEditTriggers)

        header = self.results_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)

        main_layout.addWidget(self.results_table)
        main_layout.addStretch(1)

        button_layout = QHBoxLayout()
        self.load_button = QPushButton("Lets load this up fr!")
        self.load_button.setFixedHeight(40)
        self.load_button.setStyleSheet("font-size: 16px;")
        button_layout.addWidget(self.load_button)
        self.load_button.clicked.connect(self.load_and_analyze)
        main_layout.addLayout(button_layout)

        central_widget.setLayout(main_layout)

    ##
    # @brief Loads and analyzes EEG and SpO₂ data, then displays results.
    def load_and_analyze(self):
        # --- EEG Processing ---
        from ML.Functions import load_and_filter_eeg, create_epochs, summarize_eeg_to_csv, EPOCH_DURATION, SAMPLING_RATE, stage_map, create_y_labels
        import mne

        eeg_data, raw = load_and_filter_eeg(file_path="SC4001E0-PSG.edf")
        epoch_len_in_samples = EPOCH_DURATION * SAMPLING_RATE
        eeg_epochs = create_epochs(eeg_data, epoch_len_in_samples)

        hypno_file_path = "SC4001EC-Hypnogram.edf"
        annotations = mne.read_annotations(hypno_file_path)
        raw.set_annotations(annotations)
        aligned_annotations = raw.annotations
        y_labels = create_y_labels(aligned_annotations, stage_map, EPOCH_DURATION, raw)
        summarize_eeg_to_csv(y_labels, output_csv="EEG_results.csv")

        # --- SpO₂ Processing ---
        from ML.SpOFunctions import obtain_Spo
        obtain_Spo(file_path='esp32_recorded_data.csv', channel='spo2_percent', time_channel='timestamp')

        # --- LLM Analysis ---
        from llama.llama_client import get_response
        result = get_response(eeg_csv="EEG_results.csv", spo_csv="SPO_results.csv")
        self.llm_output.setPlainText(result)
        self.display_results()

    ##
    # @brief Starts tracking window (not implemented).
    def start_tracking(self):
        print("tracking started")
        self.tracking_window = TrackingWindow()
        self.tracking_window.show()

    ##
    # @brief Ends tracking window (not implemented).
    def end_tracking(self):
        print("tracking ended")
        if hasattr(self, 'tracking_window') and self.tracking_window is not None:
            self.tracking_window.close()

    ##
    # @brief Displays results in the table widget from LLM output.
    def display_results(self):
        import csv
        from io import StringIO
        text = self.llm_output.toPlainText()
        lines = text.split('\n')
        csv_lines = []
        for line in lines:
            if ',' in line:
                csv_lines.append(line)
        if csv_lines:
            # Remove header if present
            if csv_lines and 'x' in csv_lines[0].lower():
                csv_lines = csv_lines[1:]
            self.results_table.setRowCount(0)
            for line in csv_lines:
                parts = line.split(',', 1)
                if len(parts) >= 2:
                    self.results_table.insertRow(self.results_table.rowCount())
                    self.results_table.setItem(self.results_table.rowCount()-1, 0, QTableWidgetItem(parts[0].strip()))
                    self.results_table.setItem(self.results_table.rowCount()-1, 1, QTableWidgetItem(parts[1].strip()))
