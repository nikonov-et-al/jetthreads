import numpy as np
from matplotlib import pyplot as plt
import glob
from astropy.io import fits
import library as lib

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
ax.set_xlim(20, -400)
ax.set_ylabel('Transverse offset (mas)')
ax.set_xlabel('Distance along jet (mas)')
ax.set_title('Space-Ground 3-Gaussian quasi-manual approach.')

fits_image_filename = '/home/nikonalesheo/data/M87-Radioastron/ground_max.fits'

i_map = np.squeeze(fits.getdata(fits_image_filename))
i_head = fits.getheader(fits_image_filename)

lib.imshow(i_map, i_head, fig, ax, rotation = 23, color = True, 
            lims = [[20, -450], [-80, 30]], save = 'positions', sigma_levels = 5)


fig.tight_layout()


