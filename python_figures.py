#Step 0: import libraries
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.optimize as opti

#Store the values of Time(s)/60 in liste_time_min
def time_min(liste_time_min, feuille, size):
    for i in range(size):
        liste_time_min[i] = feuille['Time_1 [s]'].iloc[i]/60
    return liste_time_min

#Store the values of df3/3 in liste_df3
def df3(liste_df3, feuille, size):
    for i in range(size):
        liste_df3[i] = (feuille['f3_1 [Hz]'].iloc[i]- feuille['f3_1 [Hz]'].iloc[0])/3
    return liste_df3

#Store the values of dD3 in liste_dD3
def dD3(liste_dD3, feuille, size):
    for i in range(size):
        liste_dD3[i] = feuille['D3_1 [ppm]'].iloc[i]-feuille['D3_1 [ppm]'].iloc[0]
    return liste_dD3

#Annotate the points with a line, arrow and text
def line_annotate(axe, liste, x, y, xtext, ytext, text):
    axe.plot([x,x], [y, liste.min()-1], color='black', linewidth=1.5, linestyle='--')
    # x,y -> where the arrow points at
    # xtext, ytext -> where the text is
    if (xtext, ytext) != (0,0):
        axe.annotate(text, xy=(x, y), xycoords='data', xytext=(xtext, ytext),
        textcoords='offset points', fontsize=10, arrowprops=dict(arrowstyle="->",
        connectionstyle="arc"))
    return

#ticks on the plot
def tick(liste_time_min, min_tick):
    last_tick = liste_time_min.max()
    tick = 30*int(last_tick/300)
    if tick:
        plt.xticks(range(0,int(last_tick),tick))
    else:
        plt.xticks(range(0,int(last_tick),min_tick))

#function to be fitted
def fit(t, a, b):
    return a*t+b
    #return -(a+b*np.log(t))
    #return -(a+b*np.sqrt(t))

#returns the values of sqrt(t) and df3/3 from 6 or 11 to 60 min
def prepare_print_datafit(liste_df3_cut, liste_time_min, liste_df3,
feuille, size, min, max):
    for i in range(size):
        if liste_time_min[i]<=11:
                # * to modify *
            min=i
        if 11 <= liste_time_min[i] <= 14:
                # * to modify *
            max=i
        if liste_time_min[i] <= 60:
                # * to modify *
            liste_df3_cut[i] = liste_df3[i]
        else:
            liste_df3_cut[i] = None
    return liste_df3_cut, min, max

def fitting_parameters(fit, liste_time_min, liste_df3_cut, min, max):
    j=0
    size = max-min

    liste_time_min_fit = np.zeros(size)
    liste_df3_cut_fit = np.zeros(size)
    for i in range(min, max):
        liste_time_min_fit[j] = liste_time_min[i]
        liste_df3_cut_fit[j] = liste_df3_cut[i]
        j=j+1

    (var,M_cov) = opti.curve_fit(fit, liste_time_min_fit, liste_df3_cut_fit)
    return (var,M_cov)

# Calcule y_théorique avec les valeurs du fit
def fit_data(y_theo, liste_df3_cut, liste_time_min, a, b, size_fit):
    for i in range(size_fit):
        if i<min:
            y_theo[i] = None
        else:
            y_theo[i] = fit(liste_time_min[i], a, b)
    return y_theo

if __name__ == "__main__":
    if len(sys.argv) < 4 :
        print("This script requires three arguments: Excel_workbook, sheet_name, plot_title")
        #ex python3 python_figures.py ~/Desktop/360_kDa_1000_ppm_PVP.xlsx Sheet1 test
        sys.exit()

    #opens sheet_name and stores everything in feuille
    feuille = pd.read_excel(sys.argv[1], sheet_name=sys.argv[2])
    title = ""
    for i in range(3, len(sys.argv)):
        title = " ".join([title, sys.argv[i]])
    #title = sys.argv[3]
    size = len(feuille)-1

    liste_time_min = np.zeros(size)
    liste_df3 = np.zeros(size)
    liste_dD3 = np.zeros(size)
    liste_df3_cut = np.zeros(size)

    #make a figure y=f(x)
    fig, axis = plt.subplots(2,1)

    #first plot, first axis ∆f3/3 = f(t)
    #fonction pour ça ?
    axis[0].plot(time_min(liste_time_min, feuille, size), df3(liste_df3, feuille,
    size), color='black',  linestyle='none', marker='s', markersize=0.5)

    axis[0].set_ylabel('∆f$_3$/3 (Hz)', fontsize=16, color='black')
    axis[0].set_xlabel('Time (min)', fontsize=16)

    # annotation coordinates for PVP
    # * to modify *
    x1, y1 = 10, 1
    xtext1, ytext1 = -40, -3

    # annotation coordinates for rinsing
    # * to modify *
    x2, y2 = 60, 1
    xtext2, ytext2 = 20, -3

    #Annotate the points with a line: e.g. 10 and 60 min
    line_annotate(axis[0], liste_df3, x1, y1, xtext1, ytext1, r'PVP')
    #line_annotate(axis[0], liste_df3, x2, y2, xtext2, ytext2, r'rinsing')

    #Draw a box on the part which is fitted below
    axis[0].plot([0, 60, 60, 0, 0], [0.5, 0.5, liste_df3.min(), liste_df3.min(), 0.5],
    color = 'blue', linewidth = 1, linestyle = '-')

    #Add an arrow pointing downwards
    axis[0].annotate("", (30, liste_df3.min()), xycoords = ('data'), xytext=(30,
    liste_df3.min()-5),
    textcoords = 'data', fontsize = 10, arrowprops = dict(arrowstyle = "<-",
    connectionstyle = "arc"))

    #first plot, second axis: ∆D3 = f(t)
    ax2 = axis[0].twinx()
    ax2.plot(time_min(liste_time_min, feuille, size), dD3(liste_dD3, feuille,
    size), color = 'red')
    ax2.set_ylabel('∆D$_3$ (ppm)', fontsize = 16, color = 'red')

    ax2.set_title(title , fontsize = 20)

    #Use Latex to set tick labels
    tick(liste_time_min, 30)

    #second plot : ∆f3/3 = f(t) with t^1/2 fit
    min, max = 0, 0
    size_fit = len(liste_df3_cut)
    y_theo = np.zeros (size_fit)

    #get indices of 11 and 60 min and df3 until 60 min
    liste_df3_cut, min, max = prepare_print_datafit(liste_df3_cut, liste_time_min,
    liste_df3, feuille, size, min, max)

    #get fitting parameters
    (var,M_cov) = fitting_parameters (fit, liste_time_min, liste_df3_cut, min, max)
    #print(M_cov)

    #get fitted data
    y_theo = fit_data(y_theo, liste_df3_cut, liste_time_min, var[0], var[1], size_fit)

    # get limite_y
    limite_y = axis[0].get_ylim()

    #plot ∆f3/3 = f(t) until 60 min
    axis[1].plot(liste_time_min, liste_df3_cut, color = 'black', linestyle = 'none',
    marker = 's', markersize = 0.5)
    axis[1].set_ylabel('∆f$_3$/3 (Hz)', fontsize = 16, color = 'black')
    axis[1].set_xlabel('Time (min)', fontsize = 16)

    #plot t^1/2 = f(t)
    ax3 = axis[1].twinx()
    # * to modify *
    #func = ['%.2f'%var[0], ' log(t) + ', '%.2f'%var[1]]
    #func = ['%.2f'%var[0], 't + ', '%2.f' %var[1]]
    func = ['%.2f'%var[0], ' t$^{1/2}$', ' + ', '%2.f' %var[1]]


    line_annotate(axis[1], liste_df3, 11, 1, 0, 0, r' ')
    line_annotate(axis[1], liste_df3, 14, 1, 0, 0, r' ')

    txt = "".join(map(str, func))
    ax3.plot(liste_time_min, y_theo, color = 'red',linestyle = '--', label = txt)

    axis[1].set_ylim(limite_y)
    ax3.set_ylim(limite_y)


    ax3.get_yaxis().set_visible(False)
    ax3.legend()
    axis[1].set_title("Adsorption model fit")

    plt.tight_layout()

    plt.show()
