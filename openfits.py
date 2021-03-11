#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 12:13:28 2018

@author: nikonalesheo

"""
from astropy.io import fits
from astropy.stats import mad_std
import numpy as np
import math
import numpy.ma as ma

import matplotlib
import matplotlib.pyplot as pic
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable


from matplotlib import rcParams

rcParams['font.family'] = 'Roboto'

FONT_SIZE = 20

matplotlib.rcParams.update({'font.size': FONT_SIZE})


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


def fits_open(filename):
    """ 
    fits_open(filename)
    
    opens fits file
    
    Keyword arguments:
        
        filename -- string -- filename of the fits file.
    
    Exmaple:
        
        fits_open('example.fits')
    
    """
    fits_file = fits.open(filename)
    image = np.squeeze(fits_file[0].data)
    header = fits_file[0].header
    return image, header

def normal_orientation(image):
    """ 
    normal_orientation(image)
    
    To show picture in normal orientation in old versions of matplotlib.
    
    Keyword arguments:
        
        image -- 2D numpy array -- image
    
    Exmaple:
        
        normal_orientation(image)
    
    """
    return image[::-1]

def ax_set(ax, im_size, scale, **kwargs):
    """
    To automaticaly set axes parameters for good looking plot.
    Sets ticks from pix to mas according to scale.
    Sets title, x and y labels.
    """
    
    
    if ('fontsize' in kwargs):
        fontsize = kwargs['fontsize']
    else:
        fontsize = FONT_SIZE
    if ('fontsize_ticks' in kwargs):
        fontsize_ticks = kwargs['fontsize_ticks']
    else:
        fontsize_ticks = FONT_SIZE
    if ('font' in kwargs):    
        font = kwargs['font']
    else:
        font = ""
#    fontsize = kwargs['fontsize'] if ('fontsize' in kwargs) else fontsize = 15
#    fontsize_ticks = kwargs['fontsize_ticks'] if kwargs['fontsize_ticks'] else fontsize = 15
    if 'title' in kwargs:  ax.set_title(kwargs['title'], 
                                         fontsize=fontsize) #fontname=font,
    if 'xlabel' in kwargs:  ax.set_xlabel(kwargs['xlabel'], 
                                         fontsize=fontsize)
    if 'ylabel' in kwargs:  ax.set_ylabel(kwargs['ylabel'],
                                         fontsize=fontsize)
    if 'background_color' in kwargs: ax.set_facecolor(kwargs['background_color'])
    
    if 'shift' in kwargs:
        main_ticks(im_size, scale, ax, fontsize = fontsize_ticks, 
                   shift = kwargs['shift'])
    else:
        main_ticks(im_size, scale, ax, fontsize = fontsize_ticks)
    
    
    

def main_ticks(image_size, scale, ax, fontsize = 15, **kwargs):
    """ 
    changes ticks to natural (miliarcseconds) from pixels of picture.
    
    Keyword arguments:
        
        image_size -- int -- image size in pixels. 
            Image should be squared.
        step -- int -- set ticks step in mas.
        scale -- float -- pixel size in mas. 
        ax -- string -- the name axes of figure (see matplotlib).
        num_comma -- int -- numbers after comma
        kwargs['shift'] --list-- [90,-50] RA, DEC
    
    Exmaple:
        
        main_ticks(image_size,10,0.08, ax)
    
    """ 
    if 'shift' in kwargs:
        offset_RA = image_size/2-kwargs['shift'][0]/scale
        offset_DEC = image_size/2+kwargs['shift'][1]/scale
    else:    
        offset_RA = image_size/2
        offset_DEC = image_size/2
    def RA_formatter (x, pos):
        s = (offset_RA-x)*scale
        if (abs(s)>1) or (abs(s) == 0.0):
            return "{:.0f}".format(s)
        else:
            return "{:.1f}".format(s)
    def DEC_formatter (x, pos):
        s = (offset_DEC-x)*scale
        if (abs(s)>1) or (abs(s) == 0.0):
            return "{:.0f}".format(s)
        else:
            return "{:.1f}".format(s)
        
    x_formatter = matplotlib.ticker.FuncFormatter(RA_formatter)
    ax.xaxis.set_major_formatter(x_formatter)
    y_formatter = matplotlib.ticker.FuncFormatter(DEC_formatter)
    ax.yaxis.set_major_formatter(y_formatter)
    
#    locator = matplotlib.ticker.LinearLocator(9)
    locator = matplotlib.ticker.LinearLocator(10)
    ax.xaxis.set_major_locator(locator)
    ax.yaxis.set_major_locator(locator)
    ax.tick_params(labelsize=fontsize)
    

def image_show(image, header, loglevs, figtitle = 'Figure',
               color_map = 'CMRmap', **kwargs):
    """ 
    shows images on the screen, saves images in png format with zoom in.
    
    Keyword arguments:
        
        image -- 2D numpy array -- the image;
        header -- header of fits image -- can be found inside fits file.
        loglevs -- float -- minimal level of image intesity to make a 
        countour (in % of intensity). 
        color_map -- string -- the name of color map (see matplotlib).
        
    Additional arguments:
    **kwargs:
        name -- string -- name one would like to save the images
    
    
    Exmaple:
        
        cut_off(image, 100, 'U')
    
    """
    scale = header['CDELT2']*60*60*1000

    image_size = image.shape[0]
    lims = [0.3,0.7,0.7,0.3]
#    lims = [0,1,1,0]
    label_font_size = '15'
    ticks_font_size = 15
    title_font_size = '10'
    background_color = 'black'
    
    hnames = {'OBJECT':'none', 'CRVAL3':0,
              'BMAJ':0,'BMIN':0,'BPA':0,'DATAMAX':0}

    if 'name' in kwargs:
        im_name = hnames['OBJECT']  
    
    
    fig, ax = pic.subplots(figsize=(19,10))
    
#    image = cut_off(image, image_size//4)
    image = normal_orientation(image)
    image_min = np.min(image)
    image_max = np.max(image)
    contours_min = loglevs
    lvls_percent = geomspace(contours_min ,1, 'true')
    lvls = lvls_percent*image_max
    im_norm = LogNorm(vmin = image_min, vmax = image_max)


    im = ax.imshow(image,  norm = im_norm , cmap = color_map )
#origin = 'lower',


    for i in hnames.keys():
        if i in header:
            hnames[i] = header[i]

    np.set_printoptions(precision=3)
    ax.set_title('Object'+' '+hnames['OBJECT']+ ' ' +str(round(
        hnames['CRVAL3']/1000000000,3))+ ' GHz ' +'\n' +' Beam FWHM: '+
        str(round(hnames['BMAJ']*60*60*1000,2))+' x'+' '+
        str(round(hnames['BMIN']*60*60*1000,2))+' mas '+'at '+
        str(round(hnames['BPA'],2))+' degrees' +'\n'+'Map peak: '+
        str(header['DATAMAX'])+'\n'+'Contours %'+str(lvls_percent*100), 
        fontsize = title_font_size)

    ax.set_xlabel('Relative RA, mas',fontsize = label_font_size)
    ax.set_ylabel('Relative DEC, mas',fontsize = label_font_size)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    cb = fig.colorbar(im,cax=cax)
    cb.ax.tick_params(labelsize = ticks_font_size)
    

    ax.contour(image, levels = lvls, colors='black', 
                          linewidths = 0.5)
    cb.set_label('Intensity, Jy/beam',fontsize = label_font_size)

  
    ax.set_facecolor(background_color)
  
    main_ticks(image_size,scale, ax)
    
    ax.set_xlim([image_size*lims[0],image_size*lims[1]])
    ax.set_ylim([image_size*lims[2],image_size*lims[3]])
    pic.rcParams['hatch.color'] = 'w'
    
    if ('BMAJ' in header) and ('BMIN' in header) and ('BPA' in header):
        ellipse = Ellipse(xy=(image_size*(lims[0]+0.1*(lims[1]-lims[0])),
                              image_size*(lims[2]-0.1*(lims[2]-lims[3]))), 
                                width=header['BMIN']*60*60*1000/scale, 
                                height=header['BMAJ']*60*60*1000/scale, 
                                angle=-header['BPA'],edgecolor = 'w', 
                                fc='gray',zorder = 0, lw=1)
        ell = ax.add_patch(ellipse)
        text1 = ax.text(image_size*(lims[0]+0.11*(lims[1]-lims[0]))+
                        header['BMAJ']*60*60*1000/scale/2,image_size*
                        (lims[2]-0.1*(lims[2]-lims[3])),
                        'Beam FWHM: \n'+str(round(header['BMAJ']*60*60*1000,2))+
                        ' x'+' '+str(round(header['BMIN']*60*60*1000,2))+
                        ' mas \n'+'at '+str(round(header['BPA'],2))+
                        ' degrees',va = 'center', color='w', fontsize='8')
 
    pic.tight_layout()
    fig.savefig(im_name+'.png',dpi = 100)
    
    fig.canvas.set_window_title(figtitle)
    
    return image, ax, fig
    

def sigma_rms(image, slice_radius):
    """
    calculates rms and sigma for image outside the source.
    
    return
        sigma, rms -- float
    """
    shape = image.shape[0]
#    N = shape**2   
#    image_max = np.max(image)

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

def cut_off(image, slice_radius):
    """ 
    cuts off image under 5*sigma level. Sigma is calculated by avaraging 
    sigmas of 4 corners of the image suggesting these regions are out of
    registered object. Values under 5*sigma are masked.
    
    Keyword arguments:
            
        image -- 2D numpy array -- the image;
        slice_radius -- int -- size in pixels of square in the corners of the image
        where sigma is calculating. 
        band -- string -- letter indicating image band: U,X... 
        
    return
        image -- 2D numpy array -- masked image.
    
    Exmaple:
        
        cut_off(image, 100, 'U')
    
    """
    shape = image.shape[0]
    N = shape**2   
    image_max = np.max(image)

    for i in [0,1,2,3]:
        if i ==0:
            cut_off_level = (5*np.sqrt(N/(N-1))*
                             np.std(image[0:slice_radius,0:slice_radius]))
        elif i ==1:
            cut_off_level = (5*np.sqrt(N/(N-1))*
                             np.std(image[shape-1-slice_radius:shape-1,
                                          0:slice_radius]) + cut_off_level)
        elif i == 2:
            cut_off_level = (5*np.sqrt(N/(N-1))*
                             np.std(image[0:slice_radius,
                                          shape-1-slice_radius:shape-1]) + 
                                                                cut_off_level)
        elif i == 3:
            cut_off_level = (5*np.sqrt(N/(N-1))*
                             np.std(image[shape-1-slice_radius:shape-1,
                                          shape-1-slice_radius:shape-1]) + 
                                                                cut_off_level)    
    cut_off_level = cut_off_level/4       
    image = ma.masked_less(image, cut_off_level)
    return image
    

pic.show()
