import numpy as np
from matplotlib import pyplot as plt
import glob

foldername = 'fit_params'
names = np.sort(glob.glob(foldername + '/fit_*.dat'))
print(names[0])
n_slices = len(names)
im_scale = 0.15

positions = [[] for i in range(n_slices)]
i = 0
for name in names:
    data = open(name, 'r')
    data_open = data.readlines()
    data_lines = (data_open[-1]).split('\t')
    data.close()
    parameters = np.array(data_lines).astype(float)
    positions[i] = parameters[1::3]
    i = i + 1

x_axis = -np.arange(0, n_slices, 1) * im_scale

fig, ax = plt.subplots(figsize = (16,4),dpi = 150)
j = 0
for x in x_axis:
    x_pos = [x for i in range(len(positions[j]))]
    ax.scatter(x_pos, positions[j], marker = '.', color = 'w')
    j = j + 1

ax.set_aspect('equal')
ax.set_ylim(-25,25)
ax.set_xlim(-20, 400)
ax.set_ylabel('Transverse offset (mas)')
ax.set_xlabel('Distance along jet (mas)')
ax.set_title('Space-Ground 3-Gaussian quasi-manual approach.')

#####################################
from matplotlib.colors import LogNorm
from scipy import ndimage
from matplotlib.offsetbox import AuxTransformBox
from matplotlib.patches import Ellipse
import openfits as op
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox
i_map, i_head = op.fits_open('/home/nikonalesheo/data/M87-Radioastron/space_min.fits')
rotation_angle = 23
i_map = ndimage.rotate(i_map, rotation_angle,reshape=True)
sigma = 1.9*10**(-4)
jet = np.ma.masked_less(i_map, sigma) ###mad_std

map_size = np.shape(i_map)


arg_max = np.unravel_index(np.argmax(i_map, axis=None), i_map.shape)
##########
#a, b - SEMI - axes of the beam, not axes. in pixels obtained from header.
a = i_head['BMAJ']*60*60*1000 / im_scale /2
b = i_head['BMIN']*60*60*1000 / im_scale /2
#k - inclination coefficient of tangent(kasatelnaya) to the ellipce y = kx + c
k = np.tan(np.deg2rad( -90 + rotation_angle - i_head['BPA']))
c = np.sqrt(a**2 * k**2 + b**2)
#beam is distance between two parallel lines of tangent
beam = abs(2*c) / (np.sqrt(k**2 + 1**2))

# Below is beam size to jet axis projetction. 
k = np.tan(np.deg2rad( -90 + rotation_angle - i_head['BPA'] + 90))
c = np.sqrt(a**2 * k**2 + b**2)
beam_jet = abs(2*c) / (np.sqrt(k**2 + 1**2))

rcParams['font.family'] = 'Roboto'

FONT_SIZE = 27

rcParams.update({'font.size': FONT_SIZE})
#########

class AnchoredEllipse(AnchoredOffsetbox):
    def __init__(self, transform, width, height, angle, loc,
        pad=0.1, borderpad=0.1, prop=None, frameon=True):
        """
        Draw an ellipse the size in data coordinate of the give axes.

        pad, borderpad in fraction of the legend font size (or prop)
        """
        self._box = AuxTransformBox(transform)
        self.ellipse = Ellipse((0, 0), width, height, angle, fc = 'grey')
        self._box.add_artist(self.ellipse)
        super().__init__(loc, pad=pad, borderpad=borderpad,
                child=self._box, prop=prop, frameon=frameon)



def draw_ellipse(ax, head):
    """
    Draw an ellipse of width=0.1, height=0.15 in data coordinates
    """
    ae = AnchoredEllipse(ax.transData,
    width=head['BMIN']*60*60*1000,
    height=head['BMAJ']*60*60*1000,
    angle=-head['BPA'] + rotation_angle,
    loc='lower left', pad=0.1, borderpad=0.1,
    frameon=True)

    ax.add_artist(ae)


def geomspace(start,finish,sqrt = None):
    """ 
    creates geometrical sequence of needed interval.

    Keyword arguments:

    start -- int -- start of sequence. Nonzero!
    finish -- int -- finish of sequence.
    sqrt = None -- don`t remember.

    Exmaple:

    geomspace(1, 100)

    """
    import math

    if sqrt == 'true':
        n = int(math.floor(1+math.log(finish/start,np.sqrt(2))))
    else:
        n = int(round(1+np.log2(finish/start)))
    y = np.empty(n)
    x = start
    i=0
    while i<n:
        y[i] = x
        x = x*np.sqrt(2)
        i = i+1
    return y


####################################

contours_min =  5*np.min(jet)/np.max(jet)
levels = geomspace(contours_min ,1, 'true')*np.max(jet)

sky_size = map_size[0]*im_scale/2

shift = (np.array([arg_max[1], arg_max[0]]) - np.array(map_size)/2) * im_scale * [1, -1]
hor_min = -(-sky_size - shift[0])
hor_max = -(sky_size - shift[0])
ver_min = -sky_size + shift[1]
ver_max = sky_size + shift[1]

ax.contour(i_map.data, levels, colors = 'black', linewidths = 0.8,  
           extent = [hor_min, hor_max, ver_min, ver_max])

ax.imshow(jet, norm = LogNorm(vmin = np.min(jet)/5, 
           vmax = np.max(jet)), cmap = 'CMRmap', origin = 'lower',                          extent = [hor_min, hor_max, ver_min, ver_max]
                   )

ax.set_xlim(20, -450)
ax.set_ylim(-50, 50)
ax.set_facecolor('black')
draw_ellipse(ax, i_head)

fig.savefig('3_gauss.pdf')




