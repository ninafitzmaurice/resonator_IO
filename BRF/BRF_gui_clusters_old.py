import sys
import torch
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QLineEdit, QSlider
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg

# TODO surrogate gradients for training 

#### SOFT RESET AND REFRACTORY PERIOD MUST BE SCALED BY OMEGA SOMEHOW ????
#### there must be a way to do this automatically and not trial and error...

#### LATER NOT IMPORTANT
# TODO spike train inputs - set % of random neurons to get input
# TODO test different spike freq inputs? need a smart way to do systematically
# TODO add distant dependent decay? 

# must fidget with a bit to get desired spiking 
R_DECAY = 0.95
b_eff_scale = 2.0

THETA = 1.0
# DUR=10.0

def gap_junction(u, W, gc=0.1):
    """
    u: (num_cells,) membrane potentials
    W: (num_cells, num_cells) adjacency matrix
    returns gap current for each cell using (u_i - u_j)
    """
    rowSum = torch.sum(W, dim=1)
    return gc * (rowSum * u - torch.matmul(W, u))


def BRF_update(x, u, v, q, b, omega, r_decay=R_DECAY, theta=THETA, dt=0.01):
    '''
    with adaptive threshold: (u_next - theta - q)
    and soft reset: b_eff = b - (q*b_eff_scale)
    '''
    # effective dampening VERY WEAK unless scaled
    b_eff = b - (q*b_eff_scale)
    # example:
    # b_eff = b - (q*10)
    u_next = u + (b_eff*u - omega*v + x)*dt
    v = v + (omega*u + b_eff*v)*dt
    # spike 
    z = ((u_next - theta - q) > 0).float()
    q = r_decay*q + z
    return z, u_next, v, q


def run_simulation(
    W,                # (num_cells x num_cells) adjacency matrix
    b_vals,           # (num_cells,) vector of betas
    omega_vals,       # (num_cells,) vector of omegas
    dur=5.0, dt=0.01,
    I_start=0.0, I_end=0.02, I_amp=10.0,
    r_decay=R_DECAY, theta=THETA, gc=0.1
):
    num_cells = W.shape[0]
    steps = int(dur / dt)
    time = torch.arange(steps) * dt

    u = torch.zeros(num_cells)
    v = torch.zeros(num_cells)
    q = torch.zeros(num_cells)

    # trajectories
    z_array = torch.zeros(num_cells, steps) 
    u_array = torch.zeros(num_cells, steps)
    v_array = torch.zeros(num_cells, steps)
    q_array = torch.zeros(num_cells, steps)

    # Input
    x_array = torch.zeros(steps)
    start_idx = int(I_start / dt)
    end_idx = int(I_end / dt)
    x_array[start_idx:end_idx] = I_amp

    ### simulation
    for i in range(steps):
        # calculate gap currents
        I_gap = gap_junction(u, W, gc)
        # all inputs 
        x_total = x_array[i] + I_gap

        z, u, v, q = BRF_update(x_total, u, v, q, b_vals, omega_vals, r_decay, theta, dt)

        z_array[:, i] = z
        u_array[:, i] = u
        v_array[:, i] = v
        q_array[:, i] = q

    return time, u_array, v_array, q_array, x_array, z_array

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("BRF Neuron Simulation (Clusters)")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()
        layout.setSpacing(4)  

        # Graph
        self.canvas = pg.GraphicsLayoutWidget()
        layout.addWidget(self.canvas)

        # RASTER PLOT FOR SPIKES
        self.rasterPlot = self.canvas.addPlot(row=0, col=0)
        self.rasterPlot.setLabel("left", "cell #")
        self.rasterPlot.hideAxis('bottom')    

        # MEM POTENTIAL AND ADAPTIVE THRESHOLD
        self.graphWidget = pg.PlotWidget()
        layout.addWidget(self.graphWidget)
        self.graphWidget.setLabel("left", "Membrane Potential (u)")
        self.graphWidget.setLabel("bottom", "Time (s)")

        # LineEdits for cluster specs:
        self.omega_input = QLineEdit("6, 12")
        self.omega_std = QLineEdit("0")

        self.beta_input = QLineEdit("-0.1, -0.1")
        self.beta_std = QLineEdit("0")

        self.cluster_sizes_input = QLineEdit("1, 1")
        # omega
        layout.addWidget(QLabel("Mean omega values (comma separated):"))
        layout.addWidget(self.omega_input)
        layout.addWidget(QLabel("Standard deviation of omega values"))
        layout.addWidget(self.omega_std)
        # beta
        layout.addWidget(QLabel("Mean beta values (comma separated):"))
        layout.addWidget(self.beta_input)
        layout.addWidget(QLabel("Standard deviation of beta values"))
        layout.addWidget(self.beta_std)
        # cluster sizes
        layout.addWidget(QLabel("Cluster sizes (comma separated):"))
        layout.addWidget(self.cluster_sizes_input)
        # gap conductance
        self.gc_input = QLineEdit("0.1") 
        layout.addWidget(QLabel("Gap Conductance:"))
        layout.addWidget(self.gc_input)
        # duration slider
        self.dur_slider = QSlider(Qt.Horizontal)
        self.dur_slider.setMinimum(1)      # 1 s
        self.dur_slider.setMaximum(20)     # 20 s
        self.dur_slider.setValue(int(5.0)) # default
        self.dur_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(QLabel("Duration (ms):"))
        layout.addWidget(self.dur_slider)


        # Update button
        self.update_button = QPushButton("Update Plot")
        self.update_button.clicked.connect(self.update_plot)
        layout.addWidget(self.update_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.update_plot()

    def parse_list(self, text):
        """
        parse comma separated floats QLineEdit
        """
        return [float(x.strip()) for x in text.split(",") if x.strip()]

    def build_adjacency(self, num_cells):
        """
        fully connected except no self-connections
        """
        W = torch.ones(num_cells, num_cells)
        # zero diagonal
        W -= torch.diag(torch.ones(num_cells))
        return W

    def update_plot(self):
        # parse user inputs
        omega_vals = self.parse_list(self.omega_input.text())  # cluster-based list
        beta_vals = self.parse_list(self.beta_input.text())
        cluster_sizes = self.parse_list(self.cluster_sizes_input.text())
        # standard deviations
        omega_std = float(self.omega_std.text())
        beta_std = float(self.beta_std.text())
        # duration
        dur = float(self.dur_slider.value())

        if len(omega_vals) != len(beta_vals) or len(omega_vals) != len(cluster_sizes):
            print("number of omega, beta, and cluster_sizes entries must match to set number of clusters")
            return

        # create the per-cell beta & omega
        b_array = []
        omega_array = []
        for i, size in enumerate(cluster_sizes):
            size_int = int(size)
            b_array     += list(torch.randn(size_int) * beta_std  + beta_vals[i])
            omega_array += list(torch.randn(size_int) * omega_std + omega_vals[i])
            # b_array.extend([beta_vals[i]] * size_int)
            # omega_array.extend([omega_vals[i]] * size_int)

        b_array = torch.tensor(b_array)
        omega_array = torch.tensor(omega_array)
        num_cells = len(b_array)

        # gap conductance
        gc = float(self.gc_input.text())

        # adjacency matrix
        W = self.build_adjacency(num_cells)

        # run
        time, u_array, v_array, q_array, x_array, z_array = run_simulation(
            W, b_array, omega_array, dur, gc=gc
        )

        # plot
        self.graphWidget.clear()
        self.rasterPlot.clear()

        self.graphWidget.setXRange(time[0], time[-1], padding=0)
        self.rasterPlot.setXLink(self.graphWidget) 
        # plot each cluster in a different color
        idx_start = 0
        colors = ['r','g','b','c','m','y','k','orange','pink','cyan']
        
        #PLOT CELLS
        for i, size in enumerate(cluster_sizes):
            size_int = int(size)
            idx_end = idx_start + size_int
            # pick color by index
            clr = colors[i % len(colors)]
            # plot average or each cell; below we plot each cell
            for cell_i in range(idx_start, idx_end):
                # plot mem potential 
                self.graphWidget.plot(time.numpy(), u_array[cell_i].numpy(), pen=pg.mkPen(clr))
                
                # adaptive threshold w refractory (dotted, on top)
                self.graphWidget.plot(
                    time.numpy(),
                    q_array[cell_i].numpy()+THETA,            
                    pen=pg.mkPen(clr, style=Qt.DotLine))
                
                # raster: times where z==1
                spike_times = time[z_array[cell_i] > 0]
                y = np.full_like(spike_times, cell_i, dtype=float)
                self.rasterPlot.addItem(
                    pg.ScatterPlotItem(spike_times.numpy(), y, pen=None,
                                    brush=pg.mkBrush(clr), size=4))
            
            idx_start = idx_end

        # input trace
        self.graphWidget.plot(time.numpy(), x_array.numpy(),
                              pen=pg.mkPen('grey', style=Qt.DashLine))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())