#Step 0: import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#stock the values of Time(s)/60 in liste_time_min
def time_min(liste_time_min, feuille, size):
    for i in range(size):
        liste_time_min[i] = feuille['Time_1 [s]'].iloc[i]/60
        i=i+1
    return liste_time_min
    #print(liste_time_min)


#stock the values of df3 in liste_df3
def df3(liste_df3, feuille, size):
    for i in range(size):
        liste_df3[i] = feuille['f3_1 [Hz]'].iloc[i]-feuille['f3_1 [Hz]'].iloc[0]
        i=i+1
    return liste_df3
    #print(liste_df3)


#stock the values of dD3 in liste_dD3
def dD3(liste_dD3, feuille, size):
    for i in range(size):
        liste_dD3[i] = feuille['D3_1 [ppm]'].iloc[i]-feuille['D3_1 [ppm]'].iloc[0]
        i=i+1
    return liste_dD3


#Annotate the points with a line, arrow and text
def line_annotate(liste, x, y, xtext, ytext, text):
    plt.plot([x,x], [y, liste.min()-1], color='black',
    linewidth=1.5, linestyle='--')
    # xy -> where the arrow points at
    # xytext -> where the text is
    plt.annotate(text, xy=(x, y), xycoords='data', xytext=(xtext, ytext),
    textcoords='offset points', fontsize=10,
    arrowprops=dict(arrowstyle="->", connectionstyle="arc"))
    return


# ticks on the plot
def tick(liste_time_min, min_tick):
    last_tick = liste_time_min.max()
    tick = 30*int(last_tick/300)
    if tick:
        plt.xticks(range(0,int(last_tick),tick))
    else:
        plt.xticks(range(0,int(last_tick),min_tick))


if __name__ == "__main__":
    # * to modify *
    #open sheet and stocks everything in feuille
    #feuille = pd.read_excel('~/Desktop/360_kDa_100_ppm_PVP.xlsx')
    #for several sheets :
    feuille = pd.read_excel('~/Desktop/360_kDa_PVP.xlsx',
    sheet_name='11000 ppm')
    size = len(feuille-1)
    # print(feuille)

    liste_time_min = np.zeros(size)
    liste_df3 = np.zeros(size)
    liste_dD3 = np.zeros(size)

    #make a figure y=f(x)
    fig, ax1 = plt.subplots()

    #first axis
    ax1.plot(time_min(liste_time_min, feuille, size), df3(liste_df3, feuille,
    size), color='black')
    ax1.set_ylabel('∆f$_3$ (Hz)', fontsize=16, color='black')

    # annotation coordinates for PVP
    # * to modify *
    x1, y1 = 5, 5
    xtext1, ytext1 = -40, -3

    # annotation coordinates for rinsing
    # * to modify *
    x2, y2 = 60, 5
    xtext2, ytext2 = 20, -3

    #Annotate the points with a line: e.g. 10 and 60 min
    line_annotate(liste_df3, x1, y1, xtext1, ytext1, r'PVP')
    line_annotate(liste_df3, x2, y2, xtext2, ytext2, r'rinsing')

    #second axis
    ax2 = ax1.twinx()
    ax2.plot(time_min(liste_time_min, feuille, size), dD3(liste_dD3, feuille,
    size), color='red')
    ax2.set_ylabel('∆D$_3$ (ppm)', fontsize=16, color='red')

    ax1.set_xlabel('Time (min)', fontsize=16)

    #Use Latex to set tick labels
    tick(liste_time_min, 30)

    #title
    # * to modify *
    plt.title('360 kDa 11000 ppm PVP', fontsize=20)

    plt.show()
