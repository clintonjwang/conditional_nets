
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Ellipse
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib as mpl
import seaborn as sns


from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count() - 2