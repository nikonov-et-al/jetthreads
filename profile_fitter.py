#Author Aleksei Nikonov
#nikonalesheo@gmail.com
#github: nikonovas

import numpy as np
import os, glob
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

### Functions ###
def gauss(x, A, x0, sigma):
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def main_plot(x, y, im_scale, profile_number,ax):
    '''
    Draws plot from the data with axis labels and title.
    '''
    global n_gauss

    ax.plot(x, y, label = 'Data')
    ax.set_xlim(-25,25)
    ax.set_xlabel('Transverse distance, mas')
    ax.set_ylabel('Intensity, Jy/beam')
    ax.set_title('Distance from the VLBI core: {:.2f} mas ({:.0f}-Gauss fit)'.format(int(profile_number)*im_scale, n_gauss) )
    ax.legend()
    return fig, ax



def multi_gauss(x, *params):
    '''
    Returns value of multi-Gausssian function.
    '''
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        A, x0, sigma = params[i:i+3]
        y = y + gauss(x, A, x0, sigma)
    return y

def multi_gauss_fitting_plotting(x, y, n_gauss, p0_prev, n_gauss_prev, ax):
    '''
    fits multi-gauss function to the data
    plots it to the axes
    '''
    global im_scale
    
    y_max = max(y)*1.2
    x_max_0 = 10
    x_max = (frame*im_scale)**0.7 + x_max_0
    
    bounds_min = ()
    bounds_max = ()
    
    for i in range(n_gauss):
        bounds_min = bounds_min + (0, -x_max, 0)
        bounds_max = bounds_max + (y_max, x_max, 10)
    
    bounds = [bounds_min, bounds_max]
    
    if n_gauss != n_gauss_prev:
        if p0_prev == []:
            p0 = [y_max/1.2, 0, 1]
            if n_gauss >= 2:
                p0 = p0 + [y_max/1.2, 5, 1]
                if n_gauss == 3:
                    p0 = p0 + [y_max/1.2, -5, 1]
        else:
            if n_gauss > n_gauss_prev:
                A_mean = sum(p0_prev[0::3])/len(p0_prev[0::3])
                x_mean = sum(p0_prev[1::3])/len(p0_prev[1::3])
                sigma_mean = sum(p0_prev[2::3])/len(p0_prev[2::3])

                if n_gauss_prev == 1:
                    if n_gauss >= 2:
                        p0 = [A_mean, x_mean - 1, sigma_mean,
                                A_mean, x_mean + 1, sigma_mean]
                        if n_gauss == 3 :
                            p0 = p0 + [A_mean, x_mean, sigma_mean]
                if n_gauss_prev == 2 and n_gauss == 3:
                    p0 = list(p0_prev) + list([A_mean, x_mean, sigma_mean])

            if n_gauss < n_gauss_prev:
                if (n_gauss == 1 and n_gauss_prev == 2) or (n_gauss == 2 and n_gauss_prev == 3):
                    I_argmin = np.argmin(p0_prev[0::3])
                    newlist = list(p0_prev.copy())
                    s = slice(I_argmin,I_argmin+3)
                    del newlist[s]
                    p0 = newlist
                elif n_gauss == 1 and n_gauss_prev == 3:
                    I_argmax = np.argmax(p0_prev[0::3])
                    newlist = p0_prev[I_argmax:I_argmax + 3].copy()
                    p0 = newlist
                

    else:
        p0 = p0_prev
        
            
    print(p0, bounds)
    popt, pcov = curve_fit(multi_gauss, x, y, 
                                       p0 = p0, 
                                       bounds = bounds,
                                       maxfev = 10000)
    
    
    
    return popt, bounds 

def multi_gauss_plot(ax, x, popt):
    '''
    Plots multi-Gaussian function with its components.
    '''
    ax.plot(x, multi_gauss(x,*popt), label = 'Fit')
    n_gauss = int(len(popt)/3)
    for i in range(0, n_gauss*3, 3):
        ax.plot(x, gauss(x, *popt[i:i+3]),
                    label = 'Gauss {:.0f}'.format(i/3 + 1))
        ax.vlines(popt[i+1], 0, popt[i])
        width = 2*np.sqrt(2*np.log(2))*popt[i+2]
        ax.hlines(popt[i]/2, -width/2 + popt[i+1],width/2 + popt[i+1]  )
    ax.legend()

def num2str1000(i, n_max):
    '''
    Returns string number in format of four digits. 
    Example: 0001, 0123...
    '''
    if i < 100 and i >= 10:
        num = '00' + str(i%n_max)
    elif i < 10:
        num = '000' + str(i)
    elif i < 1000 and i>=100:
        num = '0' + str(i)
    else:
        num = str(i)
    return num

### Functions ###


while 1:

    keyword = input('Specify the name of files contained slices like '+
                        '<<slices*.dat>> (All filenames should contain a profile number, default *dat): \n')
    if keyword == '':
        keyword = '*dat'

    file_dirs = glob.glob(keyword)
    frame_max = len(file_dirs) - 1
    
    if frame_max > -1:
        file_key = file_dirs[0][:-8]
        break
    else:
        print('Error: Files are not found. If files located in a folder in the same directory, enter path e.g. profiles/*dat\n')


im_scale = 0.15

global fig, ax
fig, ax = plt.subplots(dpi = 200)
frame = 0
n_gauss = 1
n_gauss_prev = 0

def main_loop(step):
    global manual_fit
    global frame
    global n_gauss
    global n_gauss_prev
    global im_scale
    global file_key

    manual_fit = False

    foldername = 'fit_params'
    if not foldername in os.listdir():
        os.mkdir(foldername)
    
    frame = frame + step
    print()
    filename = file_key + num2str1000(frame, frame_max) + '.dat'
    profile_number = filename[-8:-4]
    profile = np.genfromtxt(filename)
    profile_len = len(profile)
    print('Profile number: ', profile_number)
    #im_scale = 0.15

    x = np.arange( -( profile_len / 2 ), ( profile_len / 2 ), 1) * im_scale

    #main_plot(x, profile, im_scale, profile_number, ax)

    try:
        params_file = open(foldername+'/fit_' + profile_number + '.dat','r')
        params_open = params_file.readlines()
        if len(params_open) > 0:   
            params_lines = (params_open[-1]).split('\t')
            params_file.close()
            to_fit = False
            p0_prev = np.array(params_lines).astype(float)
            n_gauss = int(len(p0_prev)/3)
            print('Showed model parameters (A, x0, sigma): ', p0_prev)
            to_fit = False
        else:
            to_fit = True
            params_file.close()
            params_file = open(foldername+'/fit_' + profile_number + '.dat','w')

            try:
                previous_file = open(foldername+'/fit_' + num2str1000(int(profile_number) - 1, frame_max) + '.dat','r')
                previous_open = previous_file.readlines()
                previous_lines = (previous_open[-1]).split('\t')
                p0_prev = np.array(previous_lines).astype(float)
                n_gauss_prev = int(len(p0_prev)/3)
                
            except FileNotFoundError:
                n_gauss_prev = 0
                p0_prev = []
            
            

    except FileNotFoundError:
        if not foldername in os.listdir():
            os.mkdir(foldername)
        params_file = open(foldername+'/fit_' + profile_number + '.dat','w')
        to_fit = True
        try:
            previous_file = open(foldername+'/fit_' + num2str1000(int(profile_number) - 1, frame_max) + '.dat','r')
            previous_open = previous_file.readlines()
            previous_lines = (previous_open[-1]).split('\t')
            p0_prev = np.array(previous_lines).astype(float)
            n_gauss_prev = int(len(p0_prev)/3)
            
        except FileNotFoundError:
            n_gauss_prev = 0
            p0_prev = []
            



    if to_fit == True:
        p0_prev, bounds_prev = multi_gauss_fitting_plotting(x, profile, n_gauss, 
                                                        p0_prev,  n_gauss_prev, ax)
        
        n_gauss_prev = int(len(p0_prev)/3)
        output_line = ''
        for i in range(len(p0_prev)):
            output_line  = output_line + str(p0_prev[i]) + '\t'
        output = output_line[:-1] + '\n'
        params_file.write(output)
        params_file.close()
        print('Automatic fit parameters (A, x0, sigma): \n', p0_prev)


    main_plot(x, profile, im_scale, profile_number, ax)
    multi_gauss_plot(ax, x, p0_prev)

    global coords, gauss_comp
    coords = np.array([])

    gauss_comp = np.array([])
    #gauss_max = 0
    global params_yes
    params_yes = False

    def mouse(event):
        '''
        On users click the Gaussian forms. The coordinate of the first click
        will define an Amplitude and a x position of the Gaussian feature. 
        Second click will define the width of the Gauss.
        
        Draws vertical lines as x posotion and amplitude of Gauss function
        and horizontal lines to show the width.

        Maximum 3 Gaussians.
        '''
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #          ('double' if event.dblclick else 'single', event.button,
        #                     event.x, event.y, event.xdata, event.ydata))

        global gauss_comp
        global manual_fit
        #global n_gauss
        if (len(gauss_comp) < 3*3) and (manual_fit == True):
            global coords
            coords = np.append(coords, [event.xdata, event.ydata])
            
            if len(coords) == 2:
                ax.vlines(event.xdata, 0, event.ydata)
            elif len(coords) == 4:
                width = 2*abs(event.xdata - coords[0])
                ax.hlines(coords[1]/2, -width/2 + 
                                        coords[0], width/2 +
                                        coords[0])
                #global gauss_comp
                gauss_comp = np.append(gauss_comp, 
                          [coords[1], coords[0], 
                          width / (2*np.sqrt(2*np.log(2))) ])
                coords = np.array([])
                print('Gaussian number {:.0f} \n'.format(len(gauss_comp)/3))
            
            plt.draw()
        else:
            if len(coords) > 0:
                print('Over. Press Space to see results.')
        #fig.canvas.mpl_disconnect(mouse_connect)
            

    def key(event):
        '''
        Manual fitting managment and saving the data.
        '''
        global manual_fit
        global gauss_comp
        global fig, ax
        global params_yes
        global frame
        #print('you pressed', event.key, event.xdata, event.ydata)
        
        if event.key == ' ':
            
            if manual_fit == False:
                #global manual_fit
                #global gauss_comp
                #global coords
                print('You have chosen the manual fit.\n') 
                
                plt.cla()
                main_plot(x, profile, im_scale, profile_number, ax)
                ax.legend()
                plt.draw() 

                manual_fit = True
                gauss_comp = np.array([])
                coords = np.array([])
                print('Erasing automatic fit. Draw Gaussians by eye' + 
                        ' by clicking on the peak. Click once more to' + 
                        ' specify width of a Gaussian. \n')

            else:
                manual_fit = False
                params_yes = True
                multi_gauss_plot(ax,x,gauss_comp)
                plt.draw()
                print('You have specified {:.0f} Gaussians with corresponding'.format(len(gauss_comp)/3) + 
                        'parameters (A,x0,sigma): \n', 
                        *gauss_comp)
                print('Press Enter to write Gaussian parameters.')

        if event.key == 'a':
            global n_gauss
            global p0_prev
            #params_yes = True
            plt.cla()
            path_file = foldername+'/fit_' + profile_number + '.dat'
            if os.path.isfile(path_file): 
                os.remove(path_file) 
                main_loop(0)
                fig.canvas.mpl_disconnect(key_connect)
                fig.canvas.mpl_disconnect(mouse_connect)
                plt.draw()
                print("The profile was fitted automatically. The parameters were written to file"
                        +foldername+'/fit_' + profile_number + '.dat') 
            else: 
                print("Error: There was no fitting before. Please, restart the script.")


        if event.key == 'enter' and params_yes == True:
            params_yes = False
            output_line = ''
            for i in range(len(gauss_comp)):
                output_line  = output_line + str(gauss_comp[i]) + '\t'

            output = output_line[:-1] + '\n'
            params_file = open(foldername+'/fit_' + profile_number + '.dat','r')
            previous_lines = params_file.readlines()
            params_file.close()
            params_file = open(foldername+'/fit_' + profile_number + '.dat','w+')

            if len(previous_lines) > 1:
                params_file.writelines(previous_lines[:-1].append(output))
            else: 
                params_file.writelines(output)
            print('Saved in '+foldername+'/fit_' + profile_number + '.dat')

        if event.key == 'left':
            if frame>0:
                plt.cla()
                main_loop(-1)
                fig.canvas.mpl_disconnect(key_connect)
                fig.canvas.mpl_disconnect(mouse_connect)
                plt.draw()
        elif event.key == 'right':
            plt.cla()
            main_loop(1)
            fig.canvas.mpl_disconnect(key_connect)
            fig.canvas.mpl_disconnect(mouse_connect)
            plt.draw()

        if event.key == 'o':
            
            user_frame = input('Enter the number of the frame: ')
            try:
                frame = int(user_frame)
                plt.cla()
                main_loop(0)
                fig.canvas.mpl_disconnect(key_connect)
                fig.canvas.mpl_disconnect(mouse_connect)
                plt.draw()

            except ValueError:
                print('Please enter integer.')
            

        if event.key == '1':
            n_gauss = 1
            print('One Gaussian fitting was set.')
        elif event.key == '2':
            n_gauss = 2
            print('Two Gaussian fitting was set.')
        elif event.key == '3':
            n_gauss = 3
            print('Three Gaussian fitting was set.')
        #fig.canvas.mpl_disconnect(key_connect)

    mouse_connect = fig.canvas.mpl_connect('button_release_event', mouse)
    key_connect = fig.canvas.mpl_connect('key_release_event', key)
    if len(gauss_comp)>9:
        print('OUTPUT: ', gauss_comp)
        fig.canvas.mpl_disconnect(cid)
    #fig.canvas.mpl_disconnect(cid)
    #plt.show()
main_loop(0)
plt.show()
