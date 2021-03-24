import numpy as np
from matplotlib.colors import LogNorm
from scipy import ndimage
from matplotlib.offsetbox import AuxTransformBox
from matplotlib.patches import Ellipse
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox
from astropy.io import fits
from astropy.stats import mad_std


rcParams['font.family'] = 'Roboto'
FONT_SIZE = 27
rcParams.update({'font.size': FONT_SIZE})


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



def draw_ellipse(ax, head, rotation_angle):
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


def imshow(i_map, i_head, fig,  ax, **kwargs):
    '''
    Draws image with sqrt(2) succesive contours and logarithmic false-color intensity.

    kwargs:
    rotation --- float. In degrees, rotates image.
    color --- boolean. If False, then draws contour only image. 
    save --- string. Saves image in 'kwargs['save'].pdf'. 
    lims --- array (2,2). Array that sets limits to an image in mas. lims = [ [x_min, x_max], [y_min, y_max]  ]
    sigma_levels --- float. The lowest level to draw in Sigmas. sigma_levels = 3
    '''
    sigma = sigma_rms(i_map, 500)[0]

    if 'rotation' in kwargs:
        rotation_angle = kwargs['rotation']
        i_map = ndimage.rotate(i_map, rotation_angle, reshape=True)
        jet = np.ma.masked_less(i_map, sigma)

    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = False

    if 'sigma_levels' in kwargs:
        sigma_levels = kwargs['sigma_levels']
    else:
        sigma_levels = 1

    if 'interactive' in kwargs:
        interactive = kwargs['interactive']
    else:
        interactive = False
    
    map_size = np.shape(i_map)
    im_scale = i_head['CDELT2']*60*60*1000
    arg_max = np.unravel_index(np.argmax(i_map, axis=None), i_map.shape)
    contours_min = sigma_levels * np.min(jet)/np.max(jet)
    levels = geomspace(contours_min ,1, 'true')*np.max(jet)
    
    sky_size = map_size[0]*im_scale/2
    shift = (np.array([arg_max[1], arg_max[0]]) - np.array(map_size)/2) * im_scale * [1, -1]
    hor_min = -(-sky_size - shift[0])
    hor_max = -(sky_size - shift[0])
    ver_min = -sky_size + shift[1]
    ver_max = sky_size + shift[1]

    ax.contour(i_map.data, levels, colors = 'black', linewidths = 0.8,  
                extent = [hor_min, hor_max, ver_min, ver_max],zorder=0)
    
    draw_ellipse(ax, i_head, rotation_angle)

    if color == True:
        ax.imshow(jet, norm = LogNorm( vmin = sigma_levels * np.min(jet), vmax = np.max(jet)), 
                    cmap = 'CMRmap', origin = 'lower', 
                    extent = [hor_min, hor_max, ver_min, ver_max], zorder=0)
        ax.set_facecolor('black')
    
    if 'lims' in kwargs:
        ax.set_xlim(kwargs['lims'][0][0], kwargs['lims'][0][1])
        ax.set_ylim(kwargs['lims'][1][0], kwargs['lims'][1][1])
    
    if 'save' in kwargs:
        fig.savefig(kwargs['save'] + '.pdf')

    if interactive == True:
        plt.show()

def sigma_rms(image, slice_radius):
    """
    calculates rms and sigma for image outside the source.
    
    return
        sigma, rms -- float
    """
    shape = image.shape[0]
    
    for i in [0,1,2,3]:
        if i ==0:
            sigma = mad_std(image[0:slice_radius,0:slice_radius])
            rms = np.sqrt(np.mean(np.square(image[0:slice_radius,0:slice_radius])))
        elif i ==1:
            sigma = (mad_std(image[shape-1-slice_radius:shape-1,
                                          0:slice_radius]) + sigma)
            rms = (np.sqrt(np.mean(np.square(image[shape-1-slice_radius:shape-1,
                                          0:slice_radius]))) + rms)
        elif i == 2:
            sigma = (mad_std(image[0:slice_radius,
                                          shape-1-slice_radius:shape-1]) + 
                                                                sigma)
            rms = rms = (np.sqrt(np.mean(np.square(image[0:slice_radius,
                                          shape-1-slice_radius:shape-1]))) + rms)
        elif i == 3:
            sigma = (mad_std(image[shape-1-slice_radius:shape-1,
                                          shape-1-slice_radius:shape-1]) + 
                                                                sigma)  
            rms = (np.sqrt(np.mean(np.square(
                    image[shape-1-slice_radius:shape-1, 
                          shape-1-slice_radius:shape-1])))+rms)
                                        
    sigma = sigma/4
    rms = rms/4
    return sigma, rms




