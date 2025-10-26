from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QHeaderView, QTableWidgetItem, QTableWidget
from PyQt5.QtCore import Qt
from ui.tracking_window import TrackingWindow
from llama.llama_client import get_response


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("My PyQt5 Application")
        self.setGeometry(250, 250, 600, 400)
        self.init_ui()

    def init_ui(self):
        self.center()

    def center(self):
        qr = self.frameGeometry()
        cp = QApplication.desktop().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(15)  # slightly tighter spacing
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
        header.setSectionResizeMode(QHeaderView.Stretch)  # makes columns scale evenly
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

    def load_and_analyze(self):
        result = get_response()
        self.llm_output.setPlainText(result)
        self.display_results()

    def start_tracking(self):
        print("tracking started")
        self.tracking_window = TrackingWindow()
        self.tracking_window.show()

    def end_tracking(self):
        print("tracking ended")
        if hasattr(self, 'tracking_window') and self.tracking_window is not None:
            self.tracking_window.close()

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
