# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 12:24:55 2021

@author: Richard Cairncross and Laura Douglas
"""

# data analysis program for importing multiple datalogs

# load libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Slider
from matplotlib.widgets import Cursor

# import data files - for now put path directly in file, but in future use dialog box to browse for files
# import tkinter as tk #for dialog box
# from tkinter import filedialog

#open dialog box to  select file
# root = tk.Tk()
# root.withdraw()
# dataFile0 = filedialog.askopenfilename()
# input('hit any key to exit console')

dataFile0 = r'C:\Users\cairncra\OneDrive - drexel.edu\research\Kite Projects\DrexelKiteProjectSharedFolder\Datalogger and Analysis Codes\DataAnalysisSandbox\testData\ADATA04.CSV'
dataFile1 = r'C:\Users\cairncra\OneDrive - drexel.edu\research\Kite Projects\DrexelKiteProjectSharedFolder\Datalogger and Analysis Codes\DataAnalysisSandbox\testData\BDATA06.CSV'
dataFile2 = r'C:\Users\cairncra\OneDrive - drexel.edu\research\Kite Projects\DrexelKiteProjectSharedFolder\Datalogger and Analysis Codes\DataAnalysisSandbox\testData\CDATA04.CSV'
dataFileArray = [dataFile0, dataFile1, dataFile2]
# done: move the files into an array for automatic indexing
nFile = len(dataFileArray)

# create title for field test 
fieldTestTitle = 'Sleighton Park March 30, 2021 (RAC)'
# create titles of the dataframes to include in legends so that we can keep track of the data
dataTitle = ['Aeropod A standard', 'Kite B pocket mount', 'Aeropod C with foam damper']
dataSkipRows = [8, 7, 8]
pd.set_option('display.max_columns', None)

# import data files
# TASK - should read file line-by-line to determine how many header rows should be skipped during import
#    i += 1
df = np.empty(len(dataFileArray),dtype=object)
for i in range(len(dataFileArray)):
    df[i] = pd.read_csv(dataFileArray[i], skiprows=dataSkipRows[i])

for dfi in df:
    print(dfi.columns)

# print out statistics on the time column for each data frame
# TASK - enable user input with defaults for trim time limits
timeLabel = ['CPUt_av(s)','CPUt_av','CPUt_av(s)']
altLabel = ['AOG_av(m)','AOG_av','AOG_av(m)']

# make a plot of altitude vs time BEFORE trimming and synchronizing the files
fig1 = plt.figure(num=1, figsize = [10, 7.5])
ax1, ax2 = fig1.subplots(2, 1)


# ax1 = overview plot of all data sets vs time on the same plot
plotStyle = ['r+', 'bo', 'g*']  # define plot style for each data set
i = 0
tmin = 1000000
tmax = 0
for dfi in df:
    ax1.plot(dfi[timeLabel[i]], dfi[altLabel[i]], plotStyle[i], label = altLabel[i] + ': ' + dataTitle[i])
    ax2.plot(dfi[timeLabel[i]], dfi[altLabel[i]], plotStyle[i], label = altLabel[i] + ': ' + dataTitle[i])
    tmin = min(dfi[timeLabel[i]].min(),tmin)
    tmax = max(dfi[timeLabel[i]].max(),tmax)
    i += 1
# plt.plot(df1[timeLabel1], df1[altLabel1], 'r+', label = altLabel1 + ': ' + dataTitle[0])
# plt.plot(df2[timeLabel2], df2[altLabel2], 'bo', label = altLabel2 + ': ' + dataTitle[1])
# plt.plot(df3[timeLabel3], df3[altLabel3], 'g*', label = altLabel3 + ': ' + dataTitle[2])
#plt.plot(df1[timeLabel1], df1[altLabel1], 'r+', df2[timeLabel2], df2[altLabel2], 'bo', df3[timeLabel3], df3[altLabel3], 'g*')

# tmin = min(dfi[0][timeLabel[0]].min,dfi[1][timeLabel[1]].min,dfi[2][timeLabel[2]].min)
# tmax = max(dfi[0][timeLabel[0]].max,dfi[1][timeLabel[1]].max,dfi[2][timeLabel[2]].max)
print(tmin)
print(tmax)

# ax1 is the focus plot
ax1.grid()
ax1.set_title('FOCUS Plot of Altitude vs Sensor Time' + ': ' + fieldTestTitle)
ax1.set_xlabel('Sensor Time (s)')
ax1.set_ylabel('Altitude (m)')
ax1.set_xlim(0, 2000)
ax1.legend()
ax1.text(.5, .95, "Test of text in ax1")

# ax2 is the overview plot
ax2.grid()
ax2.set_title('OVERVIEW Plot of Altitude vs Sensor Time' + ': ' + fieldTestTitle)
ax2.set_xlabel('Sensor Time (s)')
ax2.set_ylabel('Altitude (m)')
ax2.set_xlim(0, 2000)
ax2.legend()
ax2.text(1000., 60., "Test of text in ax2")
fig1.show()

axcolor = 'lightgoldenrodyellow'
ax2span = ax2.axvspan( tmin, tmax, alpha = 0.5, facecolor = 'chocolate', lw = 5)
# ax2span.remove()
ax1.margins(x=0)

# add cursor to ax1 and ax2 Set useblit=True on most backends for enhanced performance.
cursor1 = Cursor(ax1, useblit=True, color='red', linewidth=2)
cursor2 = Cursor(ax2, useblit=True, color='red', linewidth=2)
fig1.show()

# adjust the main plot to make room for the sliders
fig1.subplots_adjust(left=0.1, bottom=0.25)

# Make a horizontal slider to control the frequency.
# TASK - make the limits of the sliders align with the limits of the overview plot)
# TASK - instead of a slider, use a vertical cursor on the overview plot to pick
#             tmin and tmax, then display tmin and tmax in a text box
axtmin = plt.axes([0.25, 0.15, 0.60, 0.03], facecolor=axcolor)
tmin_slider = Slider(
    ax=axtmin,
    label='t_min',
    valmin=tmin,
    valmax=tmax,
    valinit=tmin,
)

# adjust the main plot to make room for the sliders
#fig1.subplots_adjust(left=0.25, bottom=0.4)

# Make a horizontal slider to control the frequency.
axtmax = plt.axes([0.25, 0.05, 0.60, 0.03], facecolor=axcolor)
tmax_slider = Slider(
    ax=axtmax,
    label='t_max',
    valmin=tmin,
    valmax=tmax,
    valinit=tmax,
)

# enable updating the shading on the focus range based on the sliders
def set_xvalues(polygon, x0, x1):
    _ndarray = polygon.get_xy()
    _ndarray[:, 0] = [x0, x0, x1, x1, x0]
    polygon.set_xy(_ndarray)
    
# whenever a slider is updated, update the limits on the main figure
def update(val):
    
    focus_tmin = tmin_slider.val
    focus_tmax = tmax_slider.val
    ax1.set_xlim(focus_tmin, focus_tmax)
    #ax1.set_xlim(tmin_slider.val, tmax_slider.val)
    #TASK - add a shaded box (or vertical lines) on ax2 that shows the time
    #   limits used in the focus plot
    #ax2.axvspan.remove()
    set_xvalues(ax2span, focus_tmin, focus_tmax)
    #ax2span.remove()
    #ax2.axvspan( focus_tmin, focus_tmax, alpha = 0.5, facecolor = 'lightgoldenrodyellow', lw = 1)
    #ax2span = ax2.axvspan( focus_tmin, focus_tmax, alpha = 0.5, facecolor = 'lightgoldenrodyellow', lw = 1)
    #ax2.axvspan( tmin_slider.val, tmax_slider.val, alpha = 0.5, facecolor = 'lightgoldenrodyellow', lw = 1)
    
    print(val)
    print(tmin_slider.val)
    print(tmax_slider.val)
    print(focus_tmin)
    print(focus_tmax)    
    fig1.show()

# register the sliders
tmin_slider.on_changed(update)
tmax_slider.on_changed(update)

fig1.show()