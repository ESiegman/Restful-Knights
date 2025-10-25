from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWidgets import QPushButton




class MainWindow(QMainWindow):
  def __init__(self):
    super().__init__()
    self.setWindowTitle("My PyQt5 Application")

    self.start_button = QPushButton("Start... something lol")
    self.stop_button = QPushButton("Stop")
    self.load_button = QPushButton("Load hmm")


  def start_tracking(self):
     print("tracking started")
    #  im honstly not sure what to call this function but i thought this works for now.

  
  def end_tracking(self):
      print("tracking ended")
    #  this might end something lol.
  

  def display_results(self):
      print("displaying results")
    #  im assuming we will display some resuls here so I just put this for now.
    



