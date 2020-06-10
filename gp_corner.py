import sys
import numpy as np
from astropy.io import fits
import corner
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
from readcol import readcol
mpl.rcParams['lines.linewidth']   =3
mpl.rcParams['axes.linewidth']    = 2
mpl.rcParams['xtick.major.width'] =2
mpl.rcParams['ytick.major.width'] =2
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['legend.numpoints'] = 1
mpl.rcParams['axes.labelweight']='semibold'
mpl.rcParams['mathtext.fontset']='stix'
mpl.rcParams['font.weight'] = 'semibold'
mpl.rcParams['axes.titleweight']='semibold'
mpl.rcParams['axes.titlesize']=10
fontsize = 16
from matplotlib.pyplot import text

fits_image_filename = '../HD63433_e1.fits'
hdul = fits.open(fits_image_filename)
dat = hdul[1].data # epoch1,epoch2, Per1, Per2,rprs1, rprs2, bpar1, bpar2, rhostar, q1p1, q2p1, vsini, lambda1, q1t, q2t, intwidth, gamma1, rvtrend, rvtrendquad, sesinw1, secosw1, sesinw2, secosw2, gppparLogP, gppparLogamp, gppparLogQ1, gppparLogQ2, gppparmix


percent = 99.97

arr1 = [(np.exp(dat.gppparLogP)),((dat.gppparLogamp)),((dat.gppparLogQ2)),((dat.gppparLogQ1)),(dat.gppparmix)]
labels = [r'$P_{rot,GP}$ (days)',r'$\log(A_1)$',r'$\log(Q_0)$',r'$\log(\Delta Q)$',r'Mix']
arr2 = np.transpose(arr1)

ndim = np.shape(arr1)[0]
rng = np.zeros((ndim,2))
for i in range(0,ndim):
    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

pp = PdfPages('GP_params.pdf')
pp.savefig(fig)
pp.close()

#####

e1 = dat.secosw1**2+ dat.sesinw1**2
#arr1 = [dat.rprs1, np.abs(dat.bpar1), e1, dat.epoch1-1916, dat.Per1]
#labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$T_0$ (BJD-2458916)',r'$P$ (days)'],r'$\rho_*$']
#arr2 = np.transpose(arr1)
arr1 = [dat.rprs1, np.abs(dat.bpar1), e1, dat.rhostar, dat.Per1]
labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$\rho_*$',r'$P$ (days)']
arr1 = [dat.rprs1, np.abs(dat.bpar1), e1, dat.Per1]
labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$P$ (days)']
arr2 = np.transpose(arr1)

ndim = np.shape(arr1)[0]
rng = np.zeros((ndim,2))
for i in range(0,ndim):
    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

text(7.10855, 4620000, 'HD 63433 b',horizontalalignment='center',fontsize=18)

pp = PdfPages('Transit_b.pdf')
pp.savefig(fig)
pp.close()

#####

e2 = dat.secosw2**2+ dat.sesinw2**2
#arr1 = [dat.rprs2, np.abs(dat.bpar2), e2, dat.epoch2-1844, dat.Per2]
#labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$T_0$ (BJD-2458844)',r'$P$ (days)']
#arr2 = np.transpose(arr1)
arr1 = [dat.rprs2, np.abs(dat.bpar2), e2, dat.Per2]
labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$P$ (days)']
arr2 = np.transpose(arr1)

ndim = np.shape(arr1)[0]
rng = np.zeros((ndim,2))
for i in range(0,ndim):
    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

text(20.5453, 4500000, 'HD 63433 c',horizontalalignment='center',fontsize=18)

pp = PdfPages('Transit_c.pdf')
pp.savefig(fig)
pp.close()

#####

arr1 = [(dat.lambda1), dat.vsini, dat.gamma1+16, dat.rvtrend, dat.rvtrendquad]# dat.intwidth,
labels = [r'$\lambda$ (degrees)',r'$V\sin(i)$ (km/s)',r'$\gamma+16$ (km/s)',r'$\dot{\gamma}$ (km/s$^2$)',r'$\ddot{\gamma}$ (km/s$^3$)']#,r'width (km/s)'
arr2 = np.transpose(arr1)

ndim = np.shape(arr1)[0]
rng = np.zeros((ndim,2))
for i in range(0,ndim):
    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

pp = PdfPages('RM_params.pdf')
pp.savefig(fig)
pp.close()


#####
## matching the one in the paper

#e1 = dat.secosw1**2+ dat.sesinw1**2
#arr1 = [dat.rprs1, np.abs(dat.bpar1), e1, dat.lambda1]
#labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$\lambda$ ($^{\circ}$)']
#arr2 = np.transpose(arr1)

#ndim = np.shape(arr1)[0]
#rng = np.zeros((ndim,2))
#for i in range(0,ndim):
#    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
#fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

#
#text(0, 2500000, 'HD 63433 b',horizontalalignment='center',fontsize=18)

#pp = PdfPages('Planet1.pdf')
#pp.savefig(fig)
#pp.close()

#####

#e2 = dat.secosw2**2+ dat.sesinw2**2
#arr1 = [dat.rprs2, np.abs(dat.bpar2), e2, np.exp(dat.gppparLogP)]
#labels = [r'$R_P/R_*$',r'$b$',r'$e$',r'$P_{GP,rot}$ (days)']
#arr2 = np.transpose(arr1)

#ndim = np.shape(arr1)[0]
#rng = np.zeros((ndim,2))
#for i in range(0,ndim):
#    rng[i,:] = [np.percentile(arr2[:,i],100-percent),np.percentile(arr2[:,i],percent)]
#fig = corner.corner(arr2, fill_contours=True, plot_datapoints=False, labels=labels, show_titles=False, title_kwargs={"fontsize": fontsize},title_fmt='.4f',levels=[(1-np.exp(-0.5)),(1-np.exp(-2)),(1-np.exp(-4.5))],hist_kwargs={"linewidth": 2.5},range=rng)

#text(6.5, 100000, 'HD 63433 c',horizontalalignment='center',fontsize=18)

#pp = PdfPages('Planet2.pdf')
#pp.savefig(fig)
#pp.close()
