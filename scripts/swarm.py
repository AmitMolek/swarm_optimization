from typing import Callable, List
from PyQt5.QtCore import QObject, QThread, pyqtSignal
from PyQt5.QtGui import QIntValidator
from matplotlib.figure import Figure
import numpy as np
import sys
from dataclasses import dataclass
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, FigureCanvasAgg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import PySimpleGUI as sg
import threading
import time
from PyQt5.QtWidgets import QApplication, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QVBoxLayout, QPushButton, QWidget
import math
import sys
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt5 import QtGui
from PyQt5.QtOpenGL import *
from PyQt5 import QtCore, QtWidgets, QtOpenGL
import numpy as np

class swarm_opt:
    @dataclass
    class init_params:
        SWARM_SIZE: np.uint32 = 100
        MAX_POSITION: np.uint32 = 1
        MAX_VELOCITY: np.uint32 = 1


    @dataclass
    class swarm_params:
        W: float = 1.0
        C1: float = 0.5
        C2: float = 0.5


    def __init__(self, init_p: init_params, swarm_p: swarm_params, interest_point: np.array) -> None:
        self.__init_p = init_p
        self.__swarm_p = swarm_p

        # The particles position, initializes uniformly between [0, MAX_POSITION)
        self.__positions = np.random.uniform(low=0.0, high=init_p.MAX_POSITION, size=(init_p.SWARM_SIZE, 2))
        #self.__positions = np.random.uniform(low=-init_p.MAX_POSITION, high=init_p.MAX_POSITION, size=(init_p.SWARM_SIZE, 2))
        # The particles velocity, initializes uniformly between [0, MAX_VELOCITY)
        self.__velocities = np.random.uniform(low=-init_p.MAX_VELOCITY, high=init_p.MAX_VELOCITY, size=(init_p.SWARM_SIZE, 2))
        # Holds the best personal fitness for each particle
        self.__personal_best_fitness = np.full((init_p.SWARM_SIZE, ), sys.maxsize, dtype=np.float64)
        # Holds the best personal position for each particle
        self.__personal_best = np.full((init_p.SWARM_SIZE, 2), sys.maxsize, dtype=np.float64)
        # Holds the global best position
        self.__global_best = np.full((2,), sys.maxsize, dtype=np.float64)
        # Holds the global best fitness
        self.__global_best_fitness = sys.float_info.max
        # The point we want the particles to go to
        self.__interest_point = interest_point
        # Generation counter
        self.__generation = 0


    def __particles_fitness(self):
        return np.array([np.linalg.norm(p - self.__interest_point) for p in self.__positions])


    def set_interest_point(self, interest_point: np.array):
        self.__interest_point = interest_point


    def get_interest_point(self):
        return self.__interest_point


    def get_positions(self):
        return self.__positions


    def iterate(self, dt) -> np.array:
        self.__generation += 1

        # Calculate fitness
        fitness = self.__particles_fitness()

        # Update global fitness
        min_fitness_index = np.argmin(fitness)
        min_fitness = fitness[min_fitness_index]

        if min_fitness < self.__global_best_fitness:
            self.__global_best_fitness = min_fitness
            self.__global_best = self.__positions[min_fitness_index]

        # Update personal fitness
        personal_to_update = (fitness < self.__personal_best_fitness).nonzero()[0]
        self.__personal_best_fitness[personal_to_update] = fitness[personal_to_update]
        self.__personal_best[personal_to_update] = self.__positions[personal_to_update]

        # v(t+1) = W*v(t) + c1*rand(0,1)*(p_best-x(t)) + c2*rand(0,1)*(g_best-x(t))
        new_velocities = np.array([ \
            self.__swarm_p.W * self.__velocities[i] + \
            self.__swarm_p.C1 * np.random.rand() * (self.__personal_best[i] - x) + \
            self.__swarm_p.C2 * np.random.rand() * (self.__global_best - x) \
            for i, x in enumerate(self.__positions) \
            ])

        self.__velocities = new_velocities
        self.__positions = self.__positions + self.__velocities

        return self.__positions, self.__generation


init_params = swarm_opt.init_params(100, 100, 1)
swarm_params = swarm_opt.swarm_params(0.7, 0.4, 0.6)
interest_point = np.array([5,5])
swarm = swarm_opt(init_params, swarm_params, interest_point)

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, points_signal, parent=None):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle("Swarm Particle Optimization")
        self.widget = glWidget(points_signal)
        #self.button = QtWidgets.QPushButton('Test', self)
        mainLayout = QtWidgets.QHBoxLayout()
        mainLayout.addWidget(self.widget)
        #mainLayout.addWidget(self.button)
        self.setLayout(mainLayout)
        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.widget.update) 
        timer.start(50)


class glWidget(QGLWidget):
    def __init__(self, points_signal, parent=None):
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(640, 480)
        self.points = np.empty([0])
        self.points_signal = points_signal
        self.points_signal.signal.connect(self.__update_points)


    def __update_points(self, points):
        self.points = points


    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 0.0)
        glColor3f(0.0, 0.0, 0.0)
        glPointSize(5)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(0, 100, 0, 100)


    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)

        glColor3f(0.0, 0.0, 1.0)
        glBegin(GL_POINTS)
        glVertex2f(0,0)
        glEnd()

        glColor3f(0.0, 1.0, 0.0)
        glBegin(GL_POINTS)
        glVertex2f(5,5)
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_POINTS)
        for p in self.points:
            glVertex2f(p[0],p[1])
        glEnd()


class points_sig(QObject):
    signal = pyqtSignal(np.ndarray)

class update_points_thread(QThread):

    def __init__(self, points_signal) -> None:
        super().__init__()
        self.points_signal = points_signal


    def __del__(self):
        self.wait()


    def run(self):
        global swarm
        prev = time.time()
        while True:
            now = time.time()
            dt = now - prev

            p, g = swarm.iterate(dt)

            self.points_signal.signal.emit(p)

            prev = now
            #time.sleep(0.12)


if __name__ == '__main__':    
    app = QtWidgets.QApplication(sys.argv)
    form = QtWidgets.QMainWindow()
    points_signal = points_sig()
    ui = Ui_MainWindow(points_signal, form)
    update_thread = update_points_thread(points_signal)
    update_thread.start()
    ui.show()    
    sys.exit(app.exec_())