import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
import library as lib
from astropy.io import fits

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

fits_image_filename = '/home/nikonalesheo/data/M87-Radioastron/ground_max.fits'

i_map = np.squeeze(fits.getdata(fits_image_filename))
i_head = fits.getheader(fits_image_filename)

lib.imshow(i_map, i_head, fig, ax, rotation = 23, color = False, 
            lims = [[20, -450], [-80, 30]], save = 'new_image', sigma_levels = 5)

fig.tight_layout()

