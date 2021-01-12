#############################################################################################################################
######################################################## Global stuff #######################################################
#############################################################################################################################
import sys, os, ipywidgets, copy
from IPython.display import clear_output
if __name__ == '__main__': from global_functions import *
else: from .global_functions import *

#############################################################################################################################
###################################################### Utlity functions #####################################################
#############################################################################################################################

"""
Show the space-eigenvectors in graph from signal GFT.
--
In:
    * graph: PyGSP graph.
    * signal: 2D matrix with signals for each vertex
    * SKS: Space Kernel Scale value. Optional.
Out:
    * Interactif ipywidgets
"""
def eigenvec_anima(graph, signal, SKS = 30):
    window_kernel = create_heat_kernel (graph, SKS)
    spectrogram = compute_graph_spectrogram (graph, signal, window_kernel) 
    
    def update(instant, animation = False):
        if not animation:
            plot_graph(graph, spectrogram[instant], limits = [numpy.min(spectrogram), numpy.max(spectrogram)], 
                    title = "Frequency-lapse video representation of the spectrogram ($\lambda_{" +
                    str(instant) + "}$ = " + "{0:.4f})".format(graph.e[instant]))
        else:
            for i in range(graph.N):
                update(i); clear_output(wait = True)
    
    return ipywidgets.widgets.interact(update, instant = range(graph.N)) # Slider

#############################################################################################################################
"""
Show signals in a particular time instant.
--
In:
    * space_time_graph: List of two PyGSP graph: space(0) and time(1) graphs, in this order.
    * signal: 2D matrix with signals for each vertex
    * kwargs: Parameters of 'global_functions.plot_graph' function. Optional.
Out:
    * Interactif ipywidgets
"""
def signal_graph_anima(space_time_graph, signal, **kwargs):
    def update(instant, animation = False):
        if not animation:
            plot_graph(space_time_graph[0], signal[:, instant], 
                    limits = [numpy.min(signal), numpy.max(signal)], 
                    title = "Space-graph example at time {}".format(instant), **kwargs)
        else:
            for i in range(space_time_graph[1].N):
                update(i); clear_output(wait = True)
            animation = False
    
    return ipywidgets.widgets.interact(update, instant = range(space_time_graph[1].N)) # Slider

#############################################################################################################################
"""
Show the GFT animation, for different vertex or time-instants.
--
In:
    * graph: PyGSP graph.
    * signal: 2D matrix with signals for each vertex
    * is_graph_space: Flag to show spectogram dependent on vertex (True) or time (False). Optional.
Out:
    * Interactif ipywidgets
"""
def gft_signal_anima(graph, signal, is_graph_space = True, x_lim = None, GFT= None):
    def update(value):

        if GFT is None or not GFT :
            plot_stem(graph.igft(graph.gft(signal[value])), 
                    xticks = range(graph.N), ylabel = "Signal", title = title1 + str(value))
        if GFT is None or GFT :
            plot_stem(graph.gft(signal[value]), xticks = gft_xticks, ylabel="GFT", 
                    title = title2 + str(value),x_lim=x_lim)
    
    if is_graph_space:
        title1 = "Signal evolution at vertex "
        gft_xticks = ["$\lambda^T_{" + str(x) + "}$" for x in range(graph.N)]
        title2 = "Graph Fourier Transform (GFT) of vertex "
        description = "Vertex"
    else: # Time representation
        title1 = "Signal evolution at instance "
        gft_xticks = ["$\lambda^G_{" + str(x) + "}$" for x in range(graph.N)]
        title2 = "Graph Fourier Transform (GFT) at instance "
        description = "Instant"
        signal = signal.T

    if x_lim is not None:
        gft_xticks = gft_xticks[x_lim[0]:x_lim[1]]

    # value = ipywidgets.IntSlider(min = 0, max = signal.shape[0], step = 1, 
    #         layout = ipywidgets.Layout(width='auto'),
    #         style = {"handle_color":"lightblue"}, description = description)
    value = ipywidgets.Dropdown(options = range(signal.shape[0]), value = 0, description = description, disabled=False,
            style = {"handle_color":"lightblue"},)
    return ipywidgets.widgets.interact(update, value = value)

#############################################################################################################################
"""
Show spectogram animation, for different vertex or time-instants and dependece on space or time dimesion.
--
In:
    * graph: PyGSP graph.
    * signal: 2D matrix with signals for each vertex
    * kernel: PyGSP heat space-kernel. Optional.
    * SKS: Space Kernel Scale value. Optional.
    * is_graph_space: Flag to show spectogram dependent on vertex (True) or time (False). Optional.
    * kwargs: Parameters of 'global_functions.plot_matrix' function. Optional.
Out:
    * Interactif ipywidgets
"""
def spectogram_anima(graph, signal, kernel = None, SKS = 30, is_graph_space = False,lamdba_lim=None, **kwargs):
    def update(value, autoadj = False, interpolate = True):
        nonlocal kwargs , rows_labels
        kwargs_save = copy.deepcopy(kwargs)
        if autoadj: kwargs["limits"] = None
        kwargs["interpolate"] = interpolate
                
                
        # Space-spectogram
        
        spectrogram = compute_graph_spectrogram(graph, signal[:, value], kernel)
        if lamdba_lim is not None :
            spectrogram = spectrogram[lamdba_lim[0]:lamdba_lim[1]]
            rows_labels=rows_labels[lamdba_lim[0]:lamdba_lim[1]]
        plot_matrix(spectrogram, cols_title=cols_title, cols_labels=range(graph.N),
                    rows_title="Eigenvalue index", colorbar=True, rows_labels=rows_labels, 
                    title=title + str(value), **kwargs)
        kwargs = kwargs_save
    

    # Slider
    if is_graph_space:
        cols_title = "Vertex"
        rows_labels = ["$\lambda^G_{" + str(x) + "}$" for x in range(graph.N)]
        title = "Spatial-graph spectrogram of all values observed at time "
        desc = "Instant"
    else: # Time representation
        cols_title = "Instant"
        rows_labels = ["$\lambda^T_{" + str(x) + "}$" for x in range(graph.N)]
        title = "Time-graph spectrogram of all values observed at vertex "
        signal = signal.T
        desc = "Vertex"

    if kernel is None or SKS is not None: kernel = create_heat_kernel (graph, SKS)
    # params = {"value": ipywidgets.IntSlider(min = 0, max = signal.shape[1], step = 1, 
    #     layout = ipywidgets.Layout(width='auto'), style = {"handle_color":"lightblue"}, description = desc)}
    params = {"value": ipywidgets.Dropdown(options = range(signal.shape[1]), value = 0, 
            description = desc, disabled=False, style = {"handle_color":"lightblue"},)}
    return ipywidgets.widgets.interact(update, **params)    

#############################################################################################################################
"""
Show JFT animation, in a complex graph (space-time graphs).
--
In:
    * space_time_graph: List of two PyGSP graph: space(0) and time(1) graphs, in this order.
    * signal: 2D matrix with signals for each vertex
    * windows_kernels: List of Space Kernel Scale (SKS) and Time Kernel Scale (TKS) values, in this order. Optional.
    * kernels: Listo of PyGSP heat kernels, one for space-graph (0) and the other for time-space (1), in this order. Optional.
    * joint_spectogram: 4D matrix with joint_spectogram result. Optional.
    * kwargs: Parameters of 'global_functions.plot_matrix' function. Optional.
Out:
    * Interactif ipywidgets
"""
def JFT_anima(space_time_graph, signal, windows_kernels = None, kernels = None, joint_spectogram = None,lspace_lim=None,ltime_lim=None, **kwargs):
    def update(instant, vertex, autoadj = False, interpolate = True) :
        nonlocal kwargs
        kwargs_save = copy.deepcopy(kwargs)
        
        if autoadj: kwargs["limits"] = None
        elif "limits" not in kwargs.keys(): 
            kwargs["limits"] = [numpy.min(joint_spectogram), numpy.max(joint_spectogram)]
        kwargs["interpolate"] = interpolate
        
        Copia_JointSpectogram = joint_spectogram.copy()
        cols_labels=["$\lambda^T_{" + str(x) + "}$" for x in range(space_time_graph[1].N)]
        rows_labels=["$\lambda^G_{" + str(x) + "}$" for x in range(space_time_graph[0].N)]

        if lspace_lim is not None :
            Copia_JointSpectogram = Copia_JointSpectogram[lspace_lim[0]:lspace_lim[1]]
            rows_labels = rows_labels[lspace_lim[0]:lspace_lim[1]]
        if ltime_lim is not None :
            Copia_JointSpectogram = Copia_JointSpectogram[:,ltime_lim[0]:ltime_lim[1]]
            cols_labels = cols_labels[ltime_lim[0]:ltime_lim[1]]
        plot_matrix(Copia_JointSpectogram[:,:, vertex, instant],
                    cols_title="Eigenvalue Time-graph", rows_title="Eigenvalue Vertex-graph", colorbar=True,
                    cols_labels= cols_labels,
                    rows_labels= rows_labels ,
                    title="Graph spectrogram of all values observed at time " + 
                    str(instant) + " and vertex " + str(vertex), **kwargs)
        
        kwargs = kwargs_save
    
    # Slider
    if windows_kernels is not None:
        kernels = create_joint_heat_kernel(space_time_graph, windows_kernels)
    if kernels is not None:
        if joint_spectogram is None:
            joint_spectogram = compute_joint_graph_spectrogram(space_time_graph, signal, kernels)
    else:
        print("[ERROR] Kernels don't valid. Please check.")
        return kernels, joint_spectogram
    
    # params = {"instant": ipywidgets.IntSlider(min = 0, max = space_time_graph[1].N, step = 1, 
    #     layout = ipywidgets.Layout(width='auto'), style = {"handle_color":"lightblue"}),
    #         "vertex": ipywidgets.IntSlider(min = 0, max = space_time_graph[0].N, step = 1, 
    #     layout = ipywidgets.Layout(width='auto'), style = {"handle_color":"lightblue"})}

    params = {"instant": range(space_time_graph[1].N),
            "vertex": range(space_time_graph[0].N)}

    ipywidgets.widgets.interact(update, **params)
    return kernels, joint_spectogram

#############################################################################################################################
####################################################### Proof functions #####################################################
#############################################################################################################################

"""
Create an basic space-time graph with a random signal
--
In:
    * SGO: Space Graph Order value
    * TGO: Time Graph Order value
Out:
    * space_time_graph: List of two PyGSP graph: space(0) and time(1) graphs, in this order.
    * signal: Random signal on space_time_graph.
"""
def STG_creation(SGO, TGO):
    space_time_graph = [create_sensor_graph(SGO), create_path_graph(TGO)]
    groups = numpy.array([0]*(space_time_graph[0].N//3) + 
        [6]*(space_time_graph[0].N//3) + 
        [3]*(space_time_graph[0].N-2*space_time_graph[0].N//3))
    signal = numpy.array([space_time_graph[0].U[i, int(groups[i])] for i in range(space_time_graph[0].N)])
    signal /= numpy.linalg.norm(signal) ## Fourier function (f_hat)
    return space_time_graph, signal

if __name__ == "__main__":
    space_time_graph, signal = STG_creation(10, 10)
    eigenvec_anima(space_time_graph[0], signal, SKS = 3)    
