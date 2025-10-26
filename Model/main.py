##
# @file main.py
# @brief Entry point for the PyQt5 application.
import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow

##
# @brief Main function to start the PyQt5 application.
def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
