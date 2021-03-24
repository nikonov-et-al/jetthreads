import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

im_scale = 0.15
data_1 = np.load('thread_1.npy')
data_2 = np.load('thread_2.npy')
data_3 = np.load('thread_3.npy')

x_1 = data_1['arr_0']
y_1 = data_1['arr_1'] + 3
x_2 = data_2['arr_0']
y_2 = data_2['arr_1'] + 3 
x_3 = data_3['arr_0']
y_3 = data_3['arr_1'] + 3

def sinf(x, a, b, c):
    return a*np.sin((2*np.pi / b )*x*(1/x)**(0.56) + c) * (x/1)**(0.56)


popt_1, pcov_1 = curve_fit(sinf, x_1[50:], y_1[50:], 
                    p0 = [0.36, 5.8, np.deg2rad(250)],
                    maxfev = 20000)

popt_2, pcov_2 = curve_fit(sinf, x_2, y_2,
                    p0 = [0.36, 5.9, np.deg2rad(115)],
                    maxfev = 20000)

popt_3, pcov_3 = curve_fit(sinf, x_3[:-600], y_3[:-600],
                    p0 = [0.43, 2.6, np.deg2rad(115)],
                    maxfev = 10000)

#popt_1 = [0.45, 5.8, 115]#[-0.45, 4.6, 5269]
#popt_2 = [-0.37, 5.9, 115]
#popt_3 = [0.43, 2.5, 115] #145.8
fig, ax = plt.subplots(figsize = (18,8))

ax.scatter(-x_1,y_1 - 3, color = 'C4')#, color = 'grey')
ax.scatter(-x_2,y_2 - 3, color = 'C6')#, alpha = 0.1)#, color = 'grey')
ax.scatter(-x_3,y_3 - 3, color = 'C5')#, alpha = 0.1)#, color = 'grey')

x_1_w = np.arange(np.min(x_1), np.max(x_1), 0.15)
x_2_w = np.arange(np.min(x_2), np.max(x_2), 0.15)
x_3_w = np.arange(np.min(x_3), np.max(x_3), 0.15)

ax.plot(-x_1_w, sinf(x_1_w, *popt_1) - 3, lw = 4, color = 'C1', label = 'No 1 A = {:.2f} +- {:.2f}, Period = {:.1f} +- {:.1f} mas, Phase = {:.2f} +- {:.2f} deg.'.format(popt_1[0], np.sqrt(np.diag(pcov_1))[0],
popt_1[1], np.sqrt(np.diag(pcov_1))[1],
np.rad2deg(popt_1[2]), np.sqrt(np.diag(pcov_1))[2]) )
ax.plot(-x_1_w, sinf(x_1_w, *popt_2) - 3,lw = 4, color = 'C2', label = 'No 2 A = {:.2f} +- {:.2f}, Period = {:.1f} +- {:.1f} mas, Phase = {:.2f} +- {:.2f} deg.'.format(popt_2[0], np.sqrt(np.diag(pcov_2))[0],
popt_2[1], np.sqrt(np.diag(pcov_2))[1],
np.rad2deg(popt_2[2]), np.sqrt(np.diag(pcov_2))[2]))
x_3_w = np.arange(np.min(x_3), np.max(x_3), 0.15)
ax.plot(-x_1_w, sinf(x_1_w, *popt_3) - 3, lw = 4, color = 'C3', label = 'No 3 A = {:.2f} +- {:.2f}, Period = {:.1f} +- {:.1f} mas, Phase = {:.2f} +- {:.2f} deg.'.format(popt_3[0], np.sqrt(np.diag(pcov_3))[0],
popt_3[1], np.sqrt(np.diag(pcov_3))[1],
np.rad2deg(popt_3[2]), np.sqrt(np.diag(pcov_3))[2]))

ax.set_ylim(-60, 25)
ax.set_xlim(10, -450)
ax.set_ylabel('Transverse offset (mas)')
ax.set_xlabel('Distance along jet (mas)')
ax.set_title('Space(<100 mas) + Ground 3-Gaussian quasi-manual approach.')
ax.set_aspect('equal')
ax.legend(loc = 'lower right')

def imshow():
    from matplotlib.colors import LogNorm
    from scipy import ndimage
    from matplotlib.offsetbox import AuxTransformBox
    from matplotlib.patches import Ellipse
    import openfits as op
    from matplotlib import rcParams
    from matplotlib.offsetbox import AnchoredOffsetbox
    i_map, i_head = op.fits_open('/home/nikonalesheo/data/M87-Radioastron/ground_max.fits')
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

    contours_min =  np.min(jet)/np.max(jet)
    levels = geomspace(contours_min ,1, 'true')*np.max(jet)

    sky_size = map_size[0]*im_scale/2

    shift = (np.array([arg_max[1], arg_max[0]]) - np.array(map_size)/2) * im_scale * [1, -1]
    hor_min = -(-sky_size - shift[0])
    hor_max = -(sky_size - shift[0])
    ver_min = -sky_size + shift[1]
    ver_max = sky_size + shift[1]

    ax.contour(i_map.data, levels, colors = 'black', linewidths = 0.8,  
    extent = [hor_min, hor_max, ver_min, ver_max],zorder=0)

    #ax.imshow(jet, norm = LogNorm(vmin = np.min(jet)/5, 
    #vmax = np.max(jet)), cmap = 'CMRmap', origin = 'lower',                          extent = [hor_min, hor_max, ver_min, ver_max]
    #,zorder=0)

    ax.set_xlim(20, -450)
    #ax.set_ylim(-50, 50)
    #ax.set_facecolor('black')
    draw_ellipse(ax, i_head)

    fig.savefig('3_gauss_noend.pdf')

imshow()

fig.tight_layout()

plt.show()
