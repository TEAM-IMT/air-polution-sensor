#############################################################################################################################
######################################################## Global stuff #######################################################
#############################################################################################################################
import sys, os, ipywidgets, copy, datetime
import pandas as pd
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
def spectogram_anima(graph, signal, kernel = None, SKS = 30, is_graph_space = False, lamdba_lim = None, **kwargs):
    def update(value, autoadj = False, interpolate = True):
        nonlocal kwargs, spectrogram, pr_value
        kwargs_save = copy.deepcopy(kwargs)
        if autoadj: kwargs["limits"] = None
        kwargs["interpolate"] = interpolate

        # Space-spectogram
        if pr_value is None: pr_value = value
        if spectrogram is None or pr_value != value:
            spectrogram = compute_graph_spectrogram(graph, signal[:, value], kernel)
            pr_value = value
        if lamdba_lim is not None: 
            plot_matrix(spectrogram[lamdba_lim[0]:lamdba_lim[1]], cols_title=cols_title, cols_labels=range(graph.N),
                    rows_title="Eigenvalue index", colorbar=True, rows_labels=rows_labels[lamdba_lim[0]:lamdba_lim[1]], 
                    title=title + str(value), **kwargs)
        else:
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
    spectrogram, pr_value = None, None
    return ipywidgets.widgets.interact(update, **params)    

#############################################################################################################################
"""
Show localization kernel signal in space - graph.
--
In:
    * space_time_graph: List of two PyGSP graph: space(0) and time(1) graphs, in this order.
    * center: (vertex, instant) coordinates to localizate kernels
    * windows_kernels: List of Space Kernel Scale (SKS) and Time Kernel Scale (TKS) values, in this order. Optional.
    * kernels: Listo of PyGSP heat kernels, one for space-graph (0) and the other for time-space (1), in this order. Optional.
Out:
    * Interactif ipywidgets
"""
def kernel_anima(space_time_graph, locations, windows_kernels = None, kernels = None):
    def update(instant, autoadj = False, animaton = False):
        if not animaton:
            window_signal = signal[:,instant]
            params = {}
            if autoadj: params["limits"] = [numpy.min(window_signal), numpy.max(window_signal)]
            else: params["limits"] = [numpy.min(signal), numpy.max(signal)]
            params['title'] = "Graph with signal at (v = {}, t = {})".format(*locations) + " location and instant " + str(instant)
            plot_graph(space_time_graph[0], window_signal, **params)
        else:
            for i in range(space_time_graph[1].N):
                update(i); clear_output(wait = True)
            animation = False
    
    # Slider
    if windows_kernels is not None:
        kernels = create_joint_heat_kernel(space_time_graph, windows_kernels, normalize = True)
    if kernels is None:
        print("[ERROR] Kernels don't valid. Please check.")
        return None

    # Signal estimation
    signal = localize_joint_heat_kernel(space_time_graph, kernels, locations)

    params = {"instant": range(space_time_graph[1].N)}
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
def JFT_anima(space_time_graph, signal, windows_kernels = None, kernels = None, joint_spectogram = None,
            lspace_lim = None, ltime_lim = None, **kwargs):
    cols_labels = ["$\lambda^T_{" + str(x) + "}$" for x in range(space_time_graph[1].N)]
    rows_labels = ["$\lambda^G_{" + str(x) + "}$" for x in range(space_time_graph[0].N)]

    def update(instant, vertex, autoadj = False, interpolate = True) :
        nonlocal kwargs
        params = copy.deepcopy(kwargs)
        
        if autoadj: params["limits"] = None
        elif "limits" not in params.keys(): 
            params["limits"] = [numpy.min(joint_spectogram), numpy.max(joint_spectogram)]
        params["interpolate"] = interpolate

        params['cols_title'], params['rows_title'], params['colorbar'] = "Eigenvalue Time-graph", "Eigenvalue Vertex-graph", True
        params['title'] = "Graph spectrogram of all values observed at time " + str(instant) + " and vertex " + str(vertex)
        
        if lspace_lim is not None and ltime_lim is None:
            plot_matrix(joint_spectogram[lspace_lim[0]:lspace_lim[1], :, vertex, instant],
                cols_labels= cols_labels, rows_labels= rows_labels[lspace_lim[0]:lspace_lim[1]], **params)
        elif lspace_lim is None and ltime_lim is not None:
            plot_matrix(joint_spectogram[:, ltime_lim[0]:ltime_lim[1], vertex, instant],
                cols_labels= cols_labels[ltime_lim[0]:ltime_lim[1]], rows_labels= rows_labels, **params)
        elif lspace_lim is not None and ltime_lim is not None:
            plot_matrix(joint_spectogram[lspace_lim[0]:lspace_lim[1],ltime_lim[0]:ltime_lim[1], vertex, instant],
                cols_labels= cols_labels[ltime_lim[0]:ltime_lim[1]], rows_labels= rows_labels[lspace_lim[0]:lspace_lim[1]], **params)
        else:
            plot_matrix(joint_spectogram[:,:, vertex, instant], 
                cols_labels= cols_labels, rows_labels= rows_labels, **params)
    
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

    params = {"instant": range(space_time_graph[1].N), "vertex": range(space_time_graph[0].N)}
    ipywidgets.widgets.interact(update, **params)
    return kernels, joint_spectogram

#############################################################################################################################
"""
Show JFT animation, in a complex graph (space-time graphs), in differents time divisions.
--
In:
    * space_time_graph: List of two PyGSP graph: space(0) and time(1) graphs, in this order.
    * signal: 2D matrix with signals for each vertex
    * time_window: Time window to divide all signal in different signals
    * kwargs: Parameters of 'JFT_anima' function.
Out:
    * Interactif ipywidgets
"""
def split_JFT_anima(space_time_graph, time_sensors_series, time_window, time_div_default = 0, **kwargs):
    def update(time_division):
        nonlocal kwargs
        kwargs['kernels'], kwargs['joint_spectogram'] = None, None
        i,j = time_division*time_window, (time_division+1)*time_window
        kwargs['kernels'], 
        kwargs['joint_spectogram'] = JFT_anima([space_time_graph[0], create_path_graph(time_window)], 
                                               time_sensors_series[:,i:j], **kwargs)
    
    params = {"time_division": ipywidgets.Dropdown(options = range(space_time_graph[1].N//time_window), 
        value = time_div_default, description = 'time_division', disabled=False, style = {"handle_color":"lightblue"},)}
    ipywidgets.widgets.interact(update, **params)
    return kwargs['kernels'], kwargs['joint_spectogram']

#############################################################################################################################
"""
Plot norm_matrix, in order to detect drift concept.
--
In:
    * joint_spectogram: 4D tensor with joint spectogram values
    * sw: space window value
    * tw: time window value
Out:
    * image plot
"""

def norm_matrix_estimation(joint_spectogram, sw, tw):
    norm_matrix = numpy.zeros((joint_spectogram.shape[0] - sw,joint_spectogram.shape[1] - tw))
    for i in range(joint_spectogram.shape[0] - sw):
        for j in range(joint_spectogram.shape[1] - tw):
            norm_matrix[i,j] = numpy.linalg.norm(joint_spectogram[:,:,i:i + sw,j:j + tw])
    return norm_matrix

def plot_norm_matrix(joint_spectogram, sw, tw):
    norm_matrix = norm_matrix_estimation(joint_spectogram, sw, tw)
    plot_matrix(norm_matrix, rows_labels = ["$v_{"+str(i)+"}$" for i in range(norm_matrix.shape[0])], 
                cols_labels = ["$t_{"+str(i)+"}$" for i in range(norm_matrix.shape[1])], 
                rows_title = "Vertex", cols_title = "Instant", title = "Window spectogram normalization", colorbar = True)

#############################################################################################################################
"""
Plot dist_matrix, in order to detect drift concept.
--
In:
    * joint_spectogram: 4D tensor with joint spectogram values
    * sw: space window value
    * tw: time window value
Out:
    * image plot
"""

def dist_matrix_estimation(joint_spectogram, sw, tw, filename = None, overwrite = False, vlist = None, cprint = 100000):
    dist_matrix = pd.DataFrame()
    count, S, T = 0, *joint_spectogram.shape[2:]
    N = S*T
    joint_spectogram = numpy.pad(joint_spectogram, ((0,0),(0,0),(sw,sw),(tw,tw))) # Add zero-pad
    if filename is not None and os.path.isfile(filename) and not overwrite:
        dist_matrix = pd.read_csv(filename, index_col = 0)
        # return numpy.loadtxt(filename, delimiter = ',')
    else: 
        for a_i in range(S):
            for a_j in range(T):
                wspectogram_A = joint_spectogram[:,:, a_i:a_i + 2*sw + 1, a_j:a_j + 2*tw + 1]
                p1, dist_matrix.loc[p1,p1] = "({},{})".format(a_i,a_j), 0.0 # Distance at the same coordinate = 0
                for p2 in dist_matrix.index[:-1]:
                    b_i, b_j = eval(p2) # Get center for new spectogram
                    wspectogram_B = joint_spectogram[:,:, b_i:b_i + 2*sw + 1, b_j:b_j + 2*tw + 1]
                    dist_matrix.loc[p1,p2] = numpy.linalg.norm(wspectogram_A - wspectogram_B)
                    dist_matrix.loc[p2,p1] = dist_matrix.loc[p1,p2] # Symmetric matrix
                    if count % cprint == 0: print("[INFO] Matrix processing. {}/{}".format(count + 1, N*(N-1)//2))
                    count += 1        
        if filename is not None or overwrite: 
            # numpy.savetxt(filename, dist_matrix, delimiter = ',')
            dist_matrix.to_csv(filename)
    if vlist is not None:
        vert_list = ["({},{})".format(i,j) for i in vlist for j in range(joint_spectogram.shape[1])]
        dist_matrix = dist_matrix.loc[vert_list, vert_list]
    return dist_matrix

def plot_dist_matrix(joint_spectogram, sw, tw, vlist = None, xlim = None, ylim = None, filename = None, 
                    overwrite = False, image_save = None, npair = 200):
    print("[{}][INFO] Start matrix computation.".format(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")))
    dist_matrix = dist_matrix_estimation(joint_spectogram, sw, tw, filename = filename, overwrite = overwrite, vlist = vlist)
    print("[{}][INFO] Start matrix graph.".format(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")))
    xlabel = "" if xlim is None or (xlim[1] - xlim[0]) > npair else ["$p_{"+x+"}$" for x in dist_matrix.columns[xlim[0]:xlim[1]]]
    ylabel = "" if xlim is None or (ylim[1] - ylim[0]) > npair else ["$p_{"+x+"}$" for x in dist_matrix.index[xlim[0]:xlim[1]]]
    if xlabel == "" and len(dist_matrix.columns) <= npair: xlabel = ["$p_{"+x+"}$" for x in dist_matrix.columns]
    if ylabel == "" and len(dist_matrix.index) <= npair: ylabel = ["$p_{"+x+"}$" for x in dist_matrix.index]
    if xlim is None: xlim = [None, None]
    if ylim is None: ylim = [None, None]
    plot_matrix(dist_matrix.iloc[xlim[0]:xlim[1],ylim[0]:ylim[1]].values, rows_labels = xlabel, 
        cols_labels = "", rows_title = "Pair $(v_1,t_1)$", cols_title = "Pair $(v_2,t_2)$", 
        title = "Window spectogram Euclidian distance", colorbar = True, file_name = image_save)
    print("[{}][INFO] Final matrix computation.".format(datetime.datetime.now().strftime("%d/%m/%y %H:%M:%S")))
    return dist_matrix

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
