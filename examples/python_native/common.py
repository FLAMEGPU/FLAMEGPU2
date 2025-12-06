
import matplotlib.pyplot as plt
import threading
import time
from multiprocessing import Process, Queue
import keyboard


class LinePlotter():

    def __init__(self, title, x_label, y_label):
        self.queue = Queue()
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        # Start the plotter process
        self.p = Process(target=self._plot_process)
        self.p.daemon = True
        self.p.start()

    def add_data(self, data_point):
        self.queue.put(data_point)

    def _plot_process(self):
        plt.ion()  # non-blocking mode
        fig, ax = plt.subplots(num='FLAME GPU Live Simulation Plot')
        line, = ax.plot([], [], drawstyle="steps-post")
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        x_data = []
        y_data = []

        running = True
        while running:
            # Drain queue
            while not self.queue.empty():
                val = self.queue.get()
                x_data.append(len(x_data))
                y_data.append(val)

            # Update plot
            if x_data:
                line.set_data(x_data, y_data)
                ax.relim()
                ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.02)

