from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout,
    QWidget, QLineEdit, QSlider, QCheckBox
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import sys
import torch
import numpy as np
import sounddevice as sd      

from BRF_io import BRF_cell 
from spikeSynth import spikeSynth 

## TO DO WAY LATER: slider doesnt reset simulation, keeps same params (also sound and q toggles)
# ugh...

THETA = 1.0
DT = 0.01

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
        self.gc_input = QLineEdit("0.1") 
        self.sparse_input = QLineEdit("0.0") 
        self.q_b_params = QLineEdit("0.95, 2.0")

        # omega
        layout.addWidget(QLabel("Mean omega values (comma separated):"))
        layout.addWidget(self.omega_input)
        layout.addWidget(QLabel("Standard deviation of omega values"))
        layout.addWidget(self.omega_std)
        # beta
        layout.addWidget(QLabel("Mean beta(offset) values (comma separated):"))
        layout.addWidget(self.beta_input)
        layout.addWidget(QLabel("Standard deviation of beta(offset) values"))
        layout.addWidget(self.beta_std)
        # cluster sizes
        layout.addWidget(QLabel("Cluster sizes (comma separated):"))
        layout.addWidget(self.cluster_sizes_input)
        # gap conductance
        layout.addWidget(QLabel("Gap Conductance:"))
        layout.addWidget(self.gc_input)
        # gap junction random sparsity
        layout.addWidget(QLabel("Gap Junction Sparsity:"))
        layout.addWidget(self.sparse_input)
        # q decay and effective beta scale
        layout.addWidget(QLabel("Parameters. Q (refractory duration): | B (dampening strength):"))
        layout.addWidget(self.q_b_params)
        # duration slider
        self.dur_slider = QSlider(Qt.Horizontal)
        self.dur_slider.setMinimum(1)      # 1 s
        self.dur_slider.setMaximum(60)     # 20 s
        self.dur_slider.setValue(int(5.0)) # default
        self.dur_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(QLabel("Duration (ms):"))
        layout.addWidget(self.dur_slider)
        # show / hide refractory q
        self.q_checkbox = QCheckBox("Show adaptive threshold / refractory period")
        self.q_checkbox.setChecked(False) 
        self.q_checkbox.stateChanged.connect(self.update_plot)
        layout.addWidget(self.q_checkbox)
        # play sound
        self.synth_checkbox = QCheckBox("play spike sound")
        self.synth_checkbox.setChecked(False) 
        self.synth_checkbox.stateChanged.connect(self.update_plot)
        layout.addWidget(self.synth_checkbox)
        # update button
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

    def update_plot(self):
        # show q
        show_q = self.q_checkbox.isChecked()
        spksynth = self.synth_checkbox.isChecked()
        # parse user inputs
        omega_vals = self.parse_list(self.omega_input.text())  # cluster-based list 
        beta_vals = self.parse_list(self.beta_input.text())
        cluster_sizes = self.parse_list(self.cluster_sizes_input.text())
        # q and b params
        Q_PARAM, B_PARAM = self.parse_list(self.q_b_params.text())

        # standard deviations
        omega_std = float(self.omega_std.text())
        beta_std = float(self.beta_std.text())
        # duration
        duration = float(self.dur_slider.value())

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
        n_cells = len(b_array)

        # gap conductance
        gc = float(self.gc_input.text())
        g_sparse= float(self.sparse_input.text())
        # adjacency matrix
        W = torch.ones(n_cells,n_cells)
        W.fill_diagonal_(0.0) 

        ######################
        ### INIT BRF CELLS ###
        ######################
        IO_cells = BRF_cell(
                    W, # adjacency matrix for gap coupling
                    b_array, # beta values for simulation
                    omega_array, 
                    q_decay=Q_PARAM, # decay for refractory period / adaptive threshold / soft reset: q
                    b_eff_scale=B_PARAM, # for dampening with 
                    # effective dampening VERY WEAK unless scaled, example: test 1.0
                    theta=THETA, # threshold 
                    gc=gc, # gap conductance
                    g_sparse=g_sparse, # sparsity of gap junctions
                    dt=DT
                )
        ######################
        ### RUN SIMULATION ###
        ######################
        # current injection
        time, u_array, v_array, q_array, x_array, z_array = IO_cells.run_simulation(
                                                                dur=duration,
                                                                I_start=0.0, 
                                                                I_end=0.02, 
                                                                I_amp=10.0
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
                if show_q:
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
        
        ### SOUND!!! different pitch per cell
        ### A notes
        # synth = spikeSynth(z_array, dt_sim=DT, f_base=220.0, f_step=20.0)
        # lower pitch
        if spksynth:
            synth = spikeSynth(z_array, dt_sim=DT, f_base=110.0, f_step=5.0)
            sd.play(synth.audio, synth.Fs)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())