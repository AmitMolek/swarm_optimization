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
        self.__positions = np.random.uniform(low=-init_p.MAX_POSITION, high=init_p.MAX_POSITION, size=(init_p.SWARM_SIZE, 2))
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
        #dist_arr = np.full((SWARM_SIZE, 2), dst)
        #return np.array([np.hypot(p[0] - dst[0], p[1] - dst[1]) for p in positions])
        return np.array([np.linalg.norm(p - self.__interest_point) for p in self.__positions])


    def set_interest_point(self, interest_point: np.array):
        self.__interest_point = interest_point


    def get_interest_point(self):
        return self.__interest_point


    def get_positions(self):
        return self.__positions


    def iterate(self) -> np.array:
        self.__generation += 1

        # Calculate fitness
        fitness = self.__particles_fitness()

        # Update global fitness
        min_fitness_index = np.argmin(fitness)
        min_fitness = fitness[min_fitness_index]

        if min_fitness < self.__global_best_fitness:
            self.__global_best_fitness = min_fitness
            self.__global_best = self.__positions[min_fitness_index]

        #print(f"Global best {global_best} Global best fitness = {global_best_fitness}")

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
        self.__positions = self.__positions + new_velocities

        return self.__positions, self.__generation


class CancellationToken:
    def __init__(self) -> None:
        self.__is_cancelled = False
        self.__lock = threading.Lock()

    
    def cancel(self):
        with self.__lock:
            self.__is_cancelled = True

    
    def is_cancelled(self):
        with self.__lock:
            return self.__is_cancelled


def update_swarm(swarm: swarm_opt, ct: CancellationToken, fig, ax):
    while not ct.is_cancelled():
        swarm.iterate()


init_params = swarm_opt.init_params(100, 100, 100)
swarm_params = swarm_opt.swarm_params(0.7, 0.4, 0.6)
interest_point = np.array([5,5])
swarm = swarm_opt(init_params, swarm_params, interest_point)

'''
matplotlib.use("TkAgg")

layout = [
    [sg.InputText(str(interest_point[0]), size=(5,10), key="interest_point_x")],
    [sg.InputText(str(interest_point[1]), size=(5,10), key="interest_point_y")],
    [sg.Button("Set", key="set_interest_point")],
    [sg.Button("Start", key="start_swarm_update")],
    [sg.Button("Clear", key="clear")],
    [sg.Graph(canvas_size=(400, 400), graph_bottom_left=(0, 0), graph_top_right=(400, 400), enable_events=True, key="graph")]
]

w, h = sg.Window.get_screen_size()
window = sg.Window(
    "Matplotlib Single Graph",
    layout,
    location=(w / 4, h / 4),
    finalize=True,
    element_justification="center",
    font="Helvetica 18",
)

fig, ax = plt.subplots()

canvas = FigureCanvasTkAgg(fig, window["graph"].Widget)
plot_widget = canvas.get_tk_widget()
plot_widget.grid(row=0, column=0)
swarm_update_ct = CancellationToken()
swarm_update_thread = threading.Thread(target=update_swarm, args=(swarm, swarm_update_ct, fig, ax))
while True:
    event, values = window.read(timeout=1)
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == "set_interest_point":
        swarm.set_interest_point(np.array([int(values["interest_point_x"]), int(values["interest_point_y"])]))
    elif event == "clear":
        swarm = swarm_opt(init_params, swarm_params, interest_point)
    elif event == "start_swarm_update":
        swarm_update_thread.start()
    elif event == "stop":
        swarm_update_ct.cancel()
        swarm_update_thread.join()

    positions, gen = swarm.iterate()

    ax.cla()
    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    plt.scatter(positions[:,0], positions[:,1])
    interest_point = swarm.get_interest_point()
    plt.scatter(interest_point[0], interest_point[1])
    fig.canvas.draw()

window.close()
'''

matplotlib.use('Qt5Agg')
class mpl_canvas(FigureCanvasQTAgg):

    def __init__(self, figsize: tuple = (2.0, 2.0), dpi=100):
        figure = Figure(figsize=(2,2), dpi=dpi)
        self.axes = figure.add_subplot(111)
        super().__init__(figure)


class xy_signal(QObject):
    signal = pyqtSignal(np.ndarray, np.ndarray)


class swarm_gui(QWidget):
    def __init__(self, title: str, update_plot_sig: xy_signal) -> None:
        super().__init__()
        self.setWindowTitle(title)

        self.layout = QVBoxLayout()
        self.interest_point_layout_vbox = QVBoxLayout()
        self.layout.addLayout(self.interest_point_layout_vbox)

        self.interest_point_layout_vbox.addWidget(QLabel(text="Interest Point"))
        self.interest_point_layout = QHBoxLayout()
        self.ie_x = QLineEdit()
        self.ie_y = QLineEdit()
        self.onlyInt = QIntValidator()
        self.ie_x.setValidator(self.onlyInt)
        self.ie_y.setValidator(self.onlyInt)
        initial_ie_point = swarm.get_interest_point()
        self.ie_x.setText(str(initial_ie_point[0]))
        self.ie_y.setText(str(initial_ie_point[1]))
        self.interest_point_layout.addWidget(self.ie_x)
        self.interest_point_layout.addWidget(self.ie_y)
        self.interest_point_btn = QPushButton(text="Set")
        self.interest_point_btn.clicked.connect(self.ie_point_btn_clicked)
        self.interest_point_layout.addWidget(self.interest_point_btn)
        self.clear_swarm_btn = QPushButton(text="Start")
        self.clear_swarm_btn.clicked.connect(self.__clear_swarm)
        self.layout.addWidget(self.clear_swarm_btn)
        self.interest_point_layout_vbox.addLayout(self.interest_point_layout)

        self.sc = mpl_canvas(self)
        self.sc.axes.plot([], [])
        self.sc.axes.set_xticklabels([])
        self.sc.axes.set_yticklabels([])
        self.layout.addWidget(self.sc)

        self.setLayout(self.layout)
        self.show()

        self.update_plot_signal = update_plot_sig
        self.update_plot_signal.signal.connect(self.update_plot)


    def __clear_swarm(self):
        global swarm
        ie_point = swarm.get_interest_point()
        swarm = swarm_opt(init_params, swarm_params, ie_point)


    def get_interest_point(self):
        if self.ie_x.text() and self.ie_y.text():
            return (int(self.ie_x.text()), int(self.ie_y.text()))
        raise ValueError("EditLine is empty.")


    def ie_point_btn_clicked(self):
        global swarm
        try:
            swarm.set_interest_point(np.asarray(self.get_interest_point(), dtype=int))
        except:
            pass


    def update_plot(self, x, y):
        global swarm

        self.sc.axes.cla()
        self.sc.axes.scatter(x, y)
        ie_point = swarm.get_interest_point()
        self.sc.axes.scatter(ie_point[0], ie_point[1])
        self.sc.axes.axis("off")
        self.sc.axes.set_xlim([-100, 100])
        self.sc.axes.set_ylim([-100, 100])
        self.sc.draw()

class update_plot_thread(QThread):

    def __init__(self, xy_signal: xy_signal) -> None:
        super().__init__()
        self.signal = xy_signal


    def __del__(self):
        self.wait()


    def run(self):
        global swarm
        while True:
            p, g = swarm.iterate()
            self.signal.signal.emit(p[:,0], p[:,1])
            time.sleep(0.2)


app = QApplication([])
main_window = QMainWindow()
update_plot_signal = xy_signal()
ui = swarm_gui("Swarm Particle Optimization", update_plot_signal)
thread = update_plot_thread(update_plot_signal)
thread.start()
sys.exit(app.exec_())