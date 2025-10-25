from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QPushButton , QLabel, QVBoxLayout, QHBoxLayout, QWidget
from PyQt5.QtCore import Qt




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
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)

        title_label = QLabel("Sleep name idk")
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; margin-top: 10px;")
        main_layout.addWidget(title_label, alignment=Qt.AlignHCenter | Qt.AlignTop)

        
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("Start... something lol")
        self.stop_button = QPushButton("Stop")
        self.load_button = QPushButton("Load hmm")

        for btn in [self.start_button, self.stop_button, self.load_button]:
            btn.setFixedHeight(40)
            btn.setStyleSheet("font-size: 16px;")
            button_layout.addWidget(btn)

      
       
        main_layout.addLayout(button_layout)

        
        central_widget.setLayout(main_layout)


    def start_tracking(self):
     print("tracking started")
    #  im honstly not sure what to call this function but i thought this works for now.

  
    def end_tracking(self):
      print("tracking ended")
    #  this might end something lol.
  

    def display_results(self):
      print("displaying results")
    #  im assuming we will display some resuls here so I just put this for now.
    



