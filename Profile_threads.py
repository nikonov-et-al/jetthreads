import numpy as np
from matplotlib import pyplot as plt
import glob
from matplotlib.patches import Rectangle

foldername = 'fit_params'
names = np.sort(glob.glob(foldername + '/fit_*.dat'))
n_slices = len(names)
im_scale = 0.15

positions_y = []
i = 0

positions_x = []
for name in names:
    data = open(name, 'r')
    data_open = data.readlines()
    data_lines = (data_open[-1]).split('\t')
    data.close()
    parameters = np.array(data_lines).astype(float)
    positions_y.extend(parameters[1::3])
    x = int(name[15:-4])*im_scale
    positions_x.extend([x for i in range(len(parameters[1::3]))])
    i = i + 1

x_1 = positions_x.copy()
y_1 = positions_y.copy()
x_2 = []
y_2 = []
x_3 = []
y_3 = []

fig, ax = plt.subplots(figsize = (10, 7),dpi = 150)

lims = 0

def main_plot(ax = ax):
    global lims
    global x_1, x_2, x_3, y_1, y_2, y_3 

    ax.scatter(x_1, y_1, marker = '.', lw = 1, label = 'No 1')
    if len(x_2) >0:
        ax.scatter(x_2, y_2, marker = '.', lw = 1, label = 'No 2')
    if len(x_3) >0:
        ax.scatter(x_3, y_3, marker = '.', lw = 1, label = 'No 3')
    
    ax.set_aspect('equal')
    ax.set_ylim(-25,25)
    ax.set_xlim(-20 + lims, 100 + lims)
    ax.set_ylabel('Transverse offset (mas)')
    ax.set_xlabel('Distance along jet (mas)')
    ax.set_title('Space-Ground 3-Gaussian quasi-manual approach.')
    ax.legend()
    plt.draw()

main_plot()

coords = []
selected_x = []
selected_y = []
selection = False
thread_num = 2

def select(event):
    
    global coords
    global fig, ax
    global thread_num

    #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #          ('double' if event.dblclick else 'single', event.button,
    #event.x, event.y, event.xdata, event.ydata))
    
    if selection == True:
        if len(coords) == 0:
            ax.scatter(event.xdata, event.ydata, marker = '+')
            plt.draw()
            
        elif len(coords) == 2:
            xy = (np.min([coords[0], event.xdata]),
                    np.min([coords[1], event.ydata]))
            xy_max = (np.max([coords[0], event.xdata]),
                    np.max([coords[1], event.ydata]))
            delta_x = abs(coords[0] - event.xdata)
            delta_y = abs(coords[1] - event.ydata)
            ax.add_patch(Rectangle(xy, delta_x, delta_y,
                        edgecolor = 'black',
                        fill=False,
                        lw=2))
            plt.draw()
                    
            x_vals = np.array(x_1)
            y_vals = np.array(y_1)
            
            
            inds = np.arange(np.min( np.where(x_vals > xy[0]) ), 
                            np.max( np.where(x_vals < xy_max[0])))
            for i in inds:
                if y_vals[i] > xy[1] and y_vals[i] < xy_max[1]:
                    selected_x.append(x_vals[i])
                    selected_y.append(y_vals[i])
            for i in range(len(selected_x)):
                x_1.remove(selected_x[i])
                y_1.remove(selected_y[i])

        if len(coords) >= 4:
            coords = []
            plt.cla()
            main_plot()
        coords.extend([event.xdata, event.ydata])
          

def keyboard(event):
    global coords
    global selected_x
    global selected_y
    global selection
    global lims 
    global thread_num

    if event.key == ' ':
        print('The Selection was started.')
        selected_y = []
        selected_x = []
        selection = True
    if event.key == 'enter':
        print('The Selection was ended. All relusts were saved.')
        print('Thread No ', thread_num)
        selection = False
        coords = []
        if thread_num == 1:
            x_1.extend(selected_x)
            y_1.extend(selected_y)
        elif thread_num == 2:
            x_2.extend(selected_x) 
            y_2.extend(selected_y)
        elif thread_num == 3:
            x_3.extend(selected_x)
            y_3.extend(selected_y)

        plt.cla()
        main_plot()

    if event.key == '1':
        print('The first thread is in process.')
        thread_num = 1
    if event.key == '2':
        print('The second thread is in process.')
        thread_num = 2
    if event.key == '3':
        print('The third thread is in process.')
        thread_num = 3
    
    if event.key == 'right':
        plt.cla()
        lims = lims + 30
        main_plot()
    if event.key == 'left':
        plt.cla()
        lims = lims - 30
        main_plot()

    if event.key == 'e':
        print('All results saved to files: thread_*.npy')
        with open('thread_1.npy', 'wb') as f:
            np.savez(f, x_1, y_1)
        with open('thread_2.npy', 'wb') as f:
            np.savez(f, x_2, y_2)
        with open('thread_3.npy', 'wb') as f:
            np.savez(f, x_3, y_3)




sel_f = fig.canvas.mpl_connect('button_release_event', select)
#fig.canvas.mpl_disconnect(sel_f)

key_f = fig.canvas.mpl_connect('key_release_event', keyboard)
#fig.canvas.mpl_disconnect(key_f)


plt.show()

