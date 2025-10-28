import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtick
from matplotlib.ticker import FormatStrFormatter
from match_sfr_original import read_data
from astropy.cosmology import FlatLambdaCDM
import matplotlib.colors as colors
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import minimize
from scipy.optimize import curve_fit
from sklearn.utils import resample
from matplotlib.transforms import Affine2D
import create_simple_mock_stage1
import matplotlib
from matplotlib.lines import Line2D
matplotlib.matplotlib_fname()
import math
import h5py

def linefit(b, y, x1, x2, x3):
    y_mod = b[0] + b[1]*x1 + b[2]*x2 + b[3]*x3
    #print(np.sum((y-y_mod)**2), b)
    return np.sum((y-y_mod)**2)

def flux_func(m, band):
    if band == "g":
        m_sun, F_sun = -26.47, 4075.09 #Jy
    if band == "r":
        m_sun, F_sun = -26.93, 3254.86
    if band == "z":
        m_sun, F_sun = -27.07, 2303.28
    if band == "W1":
        m_sun, F_sun = -25.66, 314.69
    if band == "W2":
        m_sun, F_sun = -25.00, 175.16
    return F_sun*10**((m-m_sun)/-2.5)

def RGW_error_func(subvol_list, bin_edges, N, n):
    M = 1000
    RGW_sum = np.zeros((len(subvol_list), N))
    RGW_sum_2 = np.zeros((len(subvol_list), N))
    RGW_err_new = np.zeros(N)
    RGW_ave_new = np.zeros(N)
    for i in range(len(subvol_list)):
        RGW = np.load(str('properties_subvol_%d.npz' % subvol_list[i]))['y4']
        RGW_tot_old, _ = np.histogram(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6'], bins = bin_edges, weights = RGW)
        #print(len(RGW), len(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6']))
        if n == 1: #4MOST HS
            #print(len(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y9']))
            index = np.where(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']-np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y3']<0.45, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']<18))[0]
        if n == 2: #4MOST CRS
            #print(len(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y2']))
            index = np.where(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']>16, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']<18))[0]
            #print(index)
        if n == 3: #DESI BGS
            index = np.where(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] < 19.5)[0]
            #index = np.where(np.logical_and(np.logical_and(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y4'] - np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] > -1, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y4'] - np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] < 4), np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] - np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y7'] < 4, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] - np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y7'] > -1)), np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] < 19.5))[0]
        if n == 4: #DESI LIS
            index = np.where(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y4'] < 24.3, np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5'] < 23.7, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y7'] < 23.3)))[0]
        if n == 5: #LSST WFD
            index = np.where(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y0']<25.4, np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y9'] < 24.4, np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y4']<27.0, np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y5']<27.1, np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y6']<26.4, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y7']<25.2))))))[0]
            
        RGW_tot_new, _ = np.histogram(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6'][index], bins = bin_edges, weights = RGW[index])
        RGW_sum[i,:] = RGW_tot_old
        RGW_sum_2[i,:] = RGW_tot_new
    RGW_ratio_tot = np.zeros((M, N))
    RGW_vol_tot = np.zeros((M, N))
    RGW_vol_tot_2 = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            boot = np.random.choice(RGW_sum[:,j], size=10, replace=True)
            boot_2 = np.random.choice(RGW_sum_2[:,j], size=10, replace=True)
            RGW_vol_tot[i,j] = np.sum(boot)
            RGW_vol_tot_2[i,j] = np.sum(boot_2)
            RGW_ratio_tot[i,j] = RGW_vol_tot_2[i,j]/RGW_vol_tot[i,j]
    RGW_error_1 = np.zeros(N)
    RGW_err_new = np.std(RGW_ratio_tot, axis=0)
    RGW_ave_new = np.mean(RGW_ratio_tot, axis=0)
    
    return RGW_err_new, RGW_ave_new
    
if __name__ == "__main__":
    # https://arxiv.org/pdf/1304.0670.pdf
    # https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
    #H_mag, J_mag, K_mag, u_mag, g_mag, r_mag, i_mag, z_mag, deltat, lbt, redshift, all_SFH, all_MFH, z_obs, M_stellar, M_gas, SFR_obs, Z_obs, RGW = read_data(2)

    '''
    ngals = 47316
    N_Z = 180
    R_obs_all_data = np.zeros(ngals)
    R_birth_all_data = np.zeros((ngals, N_Z))
    BHFR_all_data = np.zeros((ngals, N_Z))
    BHFR_binary_data = np.zeros((ngals, N_Z))
    R_birth_binary_data = np.zeros((ngals, N_Z))
    R_merger = np.zeros((ngals, N_Z))
    
    for i in range(50):
        output_file = str("/Volumes/SeagateBackupPlusDrive/Project/Output_files_1_1/Obs_merger_rate_%d.dat.npz" % i)
        R_obs = np.load(output_file)['y1']
        R_birth = np.load(output_file)['y2']
        R_obs_all_data += R_obs
        R_birth_all_data += R_birth
    '''
    #np.savez('photometry_subvol2.npz', y0 = u_mag, y1 = J_mag, y2 = H_mag, y3 = K_mag, y4 = g_mag, y5 = r_mag, y6 = i_mag, y7 = z_mag)
    magnitudes = ['u', 'J', 'H', 'K', 'g', 'r', 'i', 'z']
    #np.savez('properties_subvolume2_1_1.npz', y1 = M_stellar, y2 = SFR_obs, y3=Z_obs, y4 = RGW)
    #data = np.load('properties_all_0.npz')
    #data_2 = np.load('properties_all_1.npz')
    #data_3 = np.load('properties_all_2.npz')
    ph_data = np.load('photometry_COMPAS_subvol_0_new.npz')
    #ph_data_2 = np.load('photometry_COMPAS_1.npz')
    #ph_data_3 = np.load('photometry_COMPAS_2.npz')
    #ph_data_4 = np.load('photometry_COMPAS_3.npz')
    data = np.load('properties_COMPAS_subvol_0_new.npz')
    #data_2 = np.load('properties_COMPAS_1.npz')
    #data_3 = np.load('properties_COMPAS_2.npz')
    #data_4 = np.load('properties_COMPAS_3.npz')
    #print(data["y4"][np.argsort(data["y4"])])
    '''
    for i in range(len(R_obs_all_data)):
        if R_obs_all_data[i] > 0:
            R_obs_new.append(R_obs_all_data[i])
            M_steller_new.append(M_stellar[i])
            Z_new.append(Z_obs[i])
    '''
    #print(np.load('/Users/Liana/Documents/PYTHON/Output_files/Obs_merger_rate_2.dat.npz')['y1'])
    cosmo = FlatLambdaCDM(H0=67.8, Om0=0.308)
    radius = cosmo.comoving_distance(1)
    radius_2 = cosmo.comoving_distance(2)
    volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius_2**3-radius**3)/3)
    f_eff = 2.15*10**-3
    #index = np.where(np.logical_and(np.logical_and( np.logical_and(ph_data['y4'] - ph_data['y5'] > -1, ph_data['y4'] - ph_data['y5'] < 4), np.logical_and(ph_data['y5'] - ph_data['y7'] < 4, ph_data['y5'] - ph_data['y7'] > -1)), ph_data['y5'] < 19.5))[0]
    #index = np.where(np.logical_and(ph_data['y1']>16, ph_data['y1']<18))[0]
    #print(len(index)*20000/(16*1.6857656436069093))
    #print(np.sum(data['y4'][index]))
    #print(f_eff*np.sum(data_3['y4'])/volume) # MERGER RATE DENSITY
    #log_M, log_SFR, log_sSFR, log_Z, log_R_obs = np.log10(M_stellar), np.log10(SFR_obs/10**9), np.log10(SFR_obs/(M_stellar*10**9)), np.log10(Z_obs), np.log10(R_obs_all_data)
    log_M, log_SFR, log_sSFR, log_Z, log_RGW = np.log10(data['y1']), np.log10(data['y2']), np.log10(data['y2']/data['y1']), np.log10(data['y3']), np.log10(data['y4'])
    #log_M_2, log_SFR_2, log_sSFR_2, log_Z_2, log_RGW_2 = np.log10(data_2['y1']), np.log10(data_2['y2']/10**9), np.log10(data_2['y2']/(data_2['y1']*10**9)), np.log10(data_2['y3']), np.log10(data_2['y4'])
    #log_M_3, log_SFR_3, log_sSFR_3, log_Z_3, log_RGW_3 = np.log10(data_3['y1']), np.log10(data_3['y2']/10**9), np.log10(data_3['y2']/(data_3['y1']*10**9)), np.log10(data_3['y3']), np.log10(data_3['y4'])
    #log_M_3, log_SFR_4, log_sSFR_4, log_Z_4, log_RGW_4 = np.log10(data_4['y1']), np.log10(data_4['y2']/10**9), np.log10(data_4['y2']/(data_4['y1']*10**9)), np.log10(data_4['y3']), np.log10(data_4['y4'])
    #print(26.1 - np.amax(ph_data['y4'][np.where(ph_data['y0'] != -999)[0]]))
    #print(np.amin(ph_data['y0'][np.where(ph_data['y0'] != -999)[0]]) - 27.4)
    beta, f_bin = 0.1, 0.7
    Normalisation_factor_numerator = (0.08**1.7/1.7 + (0.5**0.7 - 0.08**0.7)/0.7 + 0.5**-0.3/0.3)*((beta+1)*f_bin/(beta+2) - f_bin + 2)
 
    Normalisation_factor_denominator = -(200**-0.3/0.3 - 5**-0.3/0.3)*(1-0.01**(beta+1) + ((beta+1)/(beta+2))*(1-0.01**(beta+2)))
   
    Normalisation_factor = Normalisation_factor_numerator/Normalisation_factor_denominator
  
    '''
    plt.figure()
    plt.scatter(ph_data["y5"][np.where(ph_data['y5'] != -999)[0]], data["y1"][np.where(ph_data['y5'] != -999)[0]])
    plt.yscale("log")
    plt.show()
    
    fig = plt.figure()
    cmap = plt.get_cmap('gray_r')
    counts,xbins,ybins = np.histogram2d(ph_data['y0'][np.where(ph_data['y0'] != -999)[0]], ph_data['y4'][np.where(ph_data['y4'] != -999)[0]] - ph_data['y5'][np.where(ph_data['y5'] != -999)[0]],bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for l in range(len(mylevels)):
        ival = l*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))


    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))

    cax = plt.hexbin(ph_data['y0'][np.where(ph_data['y0'] != -999)[0]], ph_data['y4'][np.where(ph_data['y4'] != -999)[0]] - ph_data['y5'][np.where(ph_data['y5'] != -999)[0]], C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-5, vmax=-0.5)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", rotation=270, labelpad = 15)
    #plt.vlines(17, -1, 3.5, 'r', '--')
    #plt.ylim(-5, 3.5)
    #plt.hlines(0.45, 12, 27, 'r', '--')
    #plt.xlim(12, 27)
    #plt.fill_between([12, 17], -1, 0.45, color='grey', alpha =0.25)
    plt.xlabel('u')
    plt.ylabel('g-r')
    #plt.savefig('//Users/s4431433/Documents/codes/Paper plots/color_mag_2.png')
    plt.show()
    
    #index = np.where(np.logical_and(ph_data['y5'] - ph_data['y7'] > 0.3, ph_data['y1'] - ph_data['y3']<0.2))[0]
    index = np.where(np.logical_and(np.logical_and(log_SFR > -2.8, log_Z > -1.2), log_M > 9.6))[0]

    print(np.mean(log_SFR[index]), np.mean(log_Z[index]), np.mean(log_M[index]))
    R_sum_cutoff = np.sum(data['y4'][index])
    print(R_sum_cutoff/len(index))

    # Y = a + b1*x1 + b2*x2 http://faculty.cas.usf.edu/mbrannick/regression/Reg2IV.html
    # http://faculty.cas.usf.edu/mbrannick/regression/Part3/Reg2.html
    # Y - a - b1*x1

    #X1, X2, Y = log_M, log_SFR, log_RGW
    #b1 = (np.sum(x2**2)*np.sum(x1*y) - np.sum(x1*x2)*np.sum(x2*y))/(np.sum(x1**2)*np.sum(x2**2) - (np.sum(x1*x2))**2)
    #b2 = (np.sum(x1**2)*np.sum(x2*y) - np.sum(x1*x2)*np.sum(x1*y))/(np.sum(x1**2)*np.sum(x2**2) - (np.sum(x1*x2))**2)
    b_denominator = (np.sum(X1**2) - np.sum(X1)**2/len(X1))*(np.sum(X2**2) - np.sum(X2)**2/len(X2)) - (np.sum(X1*X2) - (np.sum(X1)*np.sum(X2))/len(X2))
    b1_numerator = (np.sum(X2**2)-(np.sum(X2)**2)/len(X2))*(np.sum(X1*Y)-(np.sum(X1)*np.sum(Y))/len(Y)) - (np.sum(X1*X2) - (np.sum(X1)*np.sum(X2))/len(X1))*(np.sum(X2*Y) - (np.sum(X2)*np.sum(Y))/len(Y))
    b2_numerator = (np.sum(X1**2)-(np.sum(X1)**2)/len(X1))*(np.sum(X2*Y)-(np.sum(X2)*np.sum(Y))/len(Y)) - (np.sum(X1*X2) - (np.sum(X1)*np.sum(X2))/len(X1))*(np.sum(X1*Y) - (np.sum(X1)*np.sum(Y))/len(Y))
    b1, b2 = b1_numerator/b_denominator, b2_numerator/b_denominator
    #b1 = ( (np.sum(X2**2)-(np.sum(X2)**2)/len(X2))*(np.sum(X1*Y)-(np.sum(X1)*np.sum(Y))/len(Y)) - (np.sum(X1)*np.sum(X2) - (np.sum(X1)*np.sum(X2))/len(X1))*(np.sum(X2)*np.sum(Y) - (np.sum(X2)*np.sum(Y))/len(Y)) ) / ( (np.sum(X1**2) - (np.sum(X1)**2)/len(X1))*(np.sum(X2**2) - (np.sum(X2**2))/len(X2)) - (np.sum(X1*X2) - (np.sum(X1)*np.sum(X2))/len(X1))**2 )
    #b2 = ( (np.sum(X1**2)-(np.sum(X1)**2)/len(X1))*(np.sum(X2*Y)-(np.sum(X2)*np.sum(Y))/len(Y)) - (np.sum(X1)*np.sum(X2) - (np.sum(X1)*np.sum(X2))/len(X1))*(np.sum(X1)*np.sum(Y) - (np.sum(X1)*np.sum(Y))/len(Y)) ) / ( (np.sum(X1**2) - (np.sum(X1)**2)/len(X1))*(np.sum(X2**2) - (np.sum(X2)**2)/len(X2)) - (np.sum(X1*X2) - (np.sum(X1)*np.sum(X2))/len(X1))**2 )
    a = np.mean(Y) - b1*np.mean(X1) - b2*np.mean(X2)
    
    #print(a, b1, b2)
    #print(log_Z.min(), log_Z.max())
    
    plt.figure()
    Z_ranges = [-5, -4.0, -3.5, -1.5, -0.5]
    colours = ['k', 'r', '#006400', 'm']
    linfit = plt.scatter(log_M, log_RGW, c=log_Z)
    cbar = plt.colorbar(linfit)
    cbar.set_label('$\log(Z)$', rotation=270, labelpad=10)
    for i in range(4):
        #index = np.where(np.logical_and(log_Z >= Z_ranges[i], log_Z < Z_ranges[i+1]))
        index = np.where(log_Z == Z_ranges[i])
        #print(len(index[0]))
        x1, x2, x3 = np.linspace(6, 12, 1000), np.linspace(-15, 0, 1000), np.linspace(-5, 0, 1000)
        #index = np.where(np.logical_and(log_Z >= -4, log_Z < -3.5))
        #plt.scatter(log_M[index], results["x"][0] + results["x"][1]*log_M[index] + results["x"][2]*log_SFR[index] , color='r')
        #index = np.where(np.logical_and(log_Z >= -3, log_Z < -2.5))
        plt.plot(x1, results["x"][0] + results["x"][1]*x1 + results["x"][2]*x2 + results["x"][3]*x3, color=str(colours[i]), label = str(Z_ranges[i]) + "<=" + "$\log(Z)$" + "<" + str(Z_ranges[i+1]))
    #plt.plot(x1, results["x"][0] + results["x"][1]*x1 + results["x"][2]*x2, color='k')
    #plt.plot(x1[index], results["x"][0] + results["x"][1]*x1+ results["x"][2]*x2 , 'b')
    #x3 = np.linspace(-2.0, -1.5, 1000)
    #plt.plot(x1, results["x"][0] + results["x"][1]*x1 + results["x"][2]*x2 + results["x"][3]*x3, 'k')
    plt.xlabel("$\log(M_* [M_\odot])$")
    plt.ylabel("$\log(R_{GW} [yr^{-1}])$")
    plt.legend(loc='upper left')
    
    #plt.show()
    #print(results["x"][0], results["x"][1], results["x"][2], results["x"][3])

    mass_ranges = [6.5, 7.5, 8.5, 9.5, 10.5]
    colours = ['b', 'r', 'g', 'm']
    for i in range(4):
        index = np.where( np.logical_and(log_M >= mass_ranges[i], log_M < mass_ranges[i+1]))
        bin_means, bin_edges, binnumber = stats.binned_statistic(log_SFR[index], log_RGW[index], statistic='mean', bins=12)
        #print(bin_edges)
        
        counts, _, _ = stats.binned_statistic(log_SFR[index], log_RGW[index], statistic='count', bins=12)
        stdev, _, _ = stats.binned_statistic(log_SFR[index], log_RGW[index], statistic='std', bins=12)
        min, _, _ = stats.binned_statistic(log_SFR[index], log_RGW[index], statistic='min', bins=12)
        max, _, _ = stats.binned_statistic(log_SFR[index], log_RGW[index], statistic='max', bins=12)
        bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])
        print(counts)
        index_count = np.where(counts > 5)
        #plt.scatter(bin_mid[index_count], bin_means[index_count], color = 'b', alpha = 0.5)
        #plt.scatter(log_SFR[index], log_RGW[index], color = 'r', alpha = 0.5)
        plt.errorbar(bin_mid[index_count], bin_means[index_count], yerr = stdev[index_count], color = str(colours[i]), label = str(mass_ranges[i]) + "<=" + "$\log(M_* [M_{\odot}])$" + "<" + str(mass_ranges[i+1]))
    #plt.errorbar(bin_mid[index_count], bin_means[index_count], yerr = max[index_count]-min[index_count], color = str(colours[i]), label = str(mass_ranges[i]) + "<=" + "log(M)" + "<" + str(mass_ranges[i+1]))
    plt.xlabel("$\log(SFR [M_{\odot}/yr])$")
    #plt.xlabel("$\log(Z)$")
    plt.ylabel("$\log(R_\mathcal{GW} [yr^{-1}])$")
    #plt.legend(loc='center left')
    plt.savefig('/Users/Liana/Documents/PYTHON/New_plots/RGW_std_SFR.pdf')
    plt.show()

    ax = plt.subplot(111)
    sfr_ranges = [-8, -6, -4, -2, -0]
    for i in range(4):
        index = np.where( np.logical_and(log_Z >= sfr_ranges[i], log_Z < sfr_ranges[i+1]))
        bin_means, bin_edges, binnumber = stats.binned_statistic(log_M[index], log_RGW[index], statistic='mean', bins=10)
        print(bin_edges)
        
        counts, _, _ = stats.binned_statistic(log_M[index], log_RGW[index], statistic='count', bins=10)
        stdev, _, _ = stats.binned_statistic(log_M[index], log_RGW[index], statistic='std', bins=10)
        bin_mid = 0.5*(bin_edges[1:] + bin_edges[:-1])
        index_count = np.where(counts > 5)
        #plt.scatter(bin_mid[index_count], bin_means[index_count], color = 'b', alpha = 0.5)
        #plt.scatter(log_SFR[index], log_RGW[index], color = 'r', alpha = 0.5)
        ax.errorbar(bin_mid[index_count], bin_means[index_count], yerr = stdev[index_count], color = str(colours[i]), label = str(sfr_ranges[i]) + "<=" + "$\log(Z)$" + "<" + str(sfr_ranges[i+1]))
    #plt.xlim(-14,6)
    ax.set_xlabel("$\log(Z)$")
    ax.set_ylabel("$\log(R_{GW} [yr^{-1}])$")
    ax.legend(loc='upper left')
    #box = ax.get_position()
    #ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    #lgd = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.savefig('/Users/Liana/Documents/PYTHON/New_plots/RGW_std_M.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.show()
    
    results = minimize(linefit, [-6.5, 0.6, 0.008, 0.01], args=(log_RGW, log_M, log_SFR, log_Z), method='Nelder-Mead', tol=1e-6)
    
    log_RGW_model = -12.445 + 1.011*log_M + 0.014*log_SFR + 0.009*log_Z
    
    #print(log_RGW_model)
    #print(log_RGW)
    #error_RGW = abs(10**log_RGW_model - data['y4'])
    #print(error_RGW)
    cmap = plt.get_cmap('gray_r')
    fig = plt.figure()
    counts,xbins,ybins = np.histogram2d(log_Z, log_SFR,bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
    #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_mean_func_2 = lambda x: np.mean(np.log10(x))
    #my_mean_func = lambda x: np.sqrt(np.sum(np.log10(x)-log_RGW_model)**2/len(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    my_sqrt_func = lambda x: np.sqrt(np.mean(x))
    #my_error_func = lambda x: np.log10(np.abs(10**log_RGW_model/np.mean(x) - 1))
    #my_chi_func = lambda x: np.sum((np.log10(x)-log_RGW_model)**2/log_RGW_model)
    #cax = plt.hexbin(log_Z, log_SFR, C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    cax2 = plt.hexbin(log_Z, log_SFR, C = data['y1'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    cax = plt.hexbin(log_Z, log_SFR, C = np.abs(10**log_RGW_model/data['y4'] - 1), gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    #cax = plt.hexbin(log_Z, log_SFR, C = (log_RGW_model-log_RGW)**2, gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_sqrt_func)
    #print(len(cax2.get_array()))
    
    binned_RGW_model = np.zeros(len(cax2.get_array()))
    bin_centers = np.array(cax.get_offsets())
    #print(cax.get_offsets())
    for j in range(len(bin_centers)):
        #print(bin_centers[j][1])
        binned_RGW_model[j] = results["x"][0] + results["x"][1]*cax2.get_array()[j] + results["x"][2]*bin_centers[j][1] + results["x"][3]*bin_centers[j][0]

    #cax = plt.hexbin(log_Z, log_SFR, gridsize=(20,20), C = data['y4'], mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    #cax2 = plt.hexbin(log_Z, log_SFR, gridsize=(20,20), C = 10**log_RGW_model, mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    #cax3 = plt.hexbin(log_Z, log_SFR, C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    
    #error = abs(binned_RGW_model/cax.get_array() - 1)
    #print(binned_RGW_model)
    #print(cax.get_array())
    #print(error)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.xlabel("$\log(Z)$", fontsize=12.5)
    plt.ylabel("$\log(\mathrm{SFR} [M_{\odot}/yr])$", fontsize=12.5)
    plt.xlim(-5, None)
    #plt.xlabel("$\log(M_* [M_{\odot}])$")
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    #cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", rotation=270, labelpad = 15)
    #cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%d}$"))
    cbar = plt.colorbar(cax)
    cbar.set_label("$\u03C3$ ($yr^{-1}$)", fontsize=12.5, rotation=270, labelpad = 18)
    #plt.savefig('/Users/Liana/Documents/PYTHON/New_plots/hexbin_ZvSFR_error.pdf')
    #plt.show()
    
    #index = np.where(log_sSFR_3>=-12)[0] #### RESIDUAL PLOT FOR SSFR CUT ####
    #log_sSFR[index] = -13 # Assume passive galaxies have min sSFR = 10**-13
    #log_M_model = np.linspace(7, 12, 100)
    #log_SFR_model = np.log10(10**-12 * 10**log_M_3[index])
    fig = plt.figure()
    cmap = plt.get_cmap('gray_r')
    counts,xbins,ybins = np.histogram2d(log_M, log_SFR,bins=30)
    #print(counts.max())
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
    #reduce_C_function=np.sum,
    
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = plt.hexbin(log_M, log_SFR, C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-9, vmax=-5)
    #plt.plot(log_M_3[index], log_SFR_model, 'r', '.')
    #plt.vlines(-1.75, -14.2, 2, 'r', '--')
    #plt.ylim(-14.2, 2)
    #plt.hlines(-2, -4.3, -0.2, 'r', '--')
    #plt.xlim(7, 12)
    #plt.fill_between([-1.75,-0.2], -2, 2, color='grey', alpha =0.25)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    #plt.xlabel("$\log(Z)$", fontsize=12.5)
    #plt.ylabel("$\log(SFR [M_{\odot}/yr])$")
    plt.xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    plt.ylabel("$\log(\mathrm{SFR} [\mathrm{M_{\odot}/yr}])$", fontsize=12.5)
    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    cbar.set_label("Average $R_\mathrm{GW, obs} [\mathrm{yr^{-1}}]$", fontsize=12.5, rotation=270, labelpad = 18)
    #plt.savefig('/Users/s4431433/Documents/codes/Paper plots subvolume 1/hexbin_SFRvsZ_mean.png')
    plt.show()
    
    x1, x2, x3 = np.linspace(6,12,100), np.linspace(-6,3,100), np.linspace(-6,0,100)
    results = minimize(linefit, [-5, 0.8, 0.5, 0.01], args=(log_RGW_3[index], log_M_3[index], log_SFR_3[index], log_Z_3[index]), method='Nelder-Mead', tol=1e-6)
    log_RGW_model = results["x"][0] + results["x"][1]*log_M_3[index] + results["x"][2]*log_SFR_3[index] + results["x"][3]*log_Z_3[index]
    print(results["x"][0], results["x"][1], results["x"][2], results["x"][3])
    lin = np.linspace(-10, 0, 100)
    plt.scatter(log_RGW_model, log_RGW_3[index], s=2, c='red')
    plt.plot(lin, lin, 'k')
    plt.xlabel("Predicted $R_{GW}$")
    plt.ylabel("Actual $R_{GW}$")
    plt.xlim(-5, 0)
    plt.show()
    
    fig = plt.figure()
    cmap = plt.get_cmap('gray_r')
    counts,xbins,ybins = np.histogram2d(log_Z, log_SFR,bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
    #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = plt.hexbin(log_Z, log_SFR, C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-9, vmax=-5)
    #plt.hlines(-2.8, -5, -0.22, 'r', '--')
    #plt.vlines(-1.3, -14.27, 1.95, 'r', '--')
    #plt.xlim(-5, -0.22)
    #plt.ylim(-14.27, 1.95)
    #plt.fill_between([-1.3, -0.22], -2.8, 1.95, color='grey', alpha =0.25)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.xlabel("$\log(Z)$", fontsize=12.5)
    plt.ylabel("$\log(\mathrm{SFR} [\mathrm{M_{\odot}/yr}])$", fontsize=12.5)
    #plt.xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%d}$"))
    cbar.set_label("Average $R_\mathrm{GW, obs} [\mathrm{yr^{-1}}]$", fontsize=12.5, rotation=270, labelpad = 18)
    #plt.savefig('/Users/Liana/Documents/PYTHON/New_plots/hexbin_ZvSFR_sum.pdf')
    plt.show()
    
    
    zobs = data['y5']
    
    fig, axs = plt.subplots(nrows=1, ncols=3)
    ax = axs[0]
    log_RGW_new = log_RGW[np.where(np.logical_and(zobs>0, zobs<=1))[0]]
    log_M_new = log_M[np.where(np.logical_and(zobs>0, zobs<=1))[0]]
    log_SFR_new = log_SFR[np.where(np.logical_and(zobs>0, zobs<=1))[0]]
    log_Z_new = log_Z[np.where(np.logical_and(zobs>0, zobs<=1))[0]]
    
    x1, x2, x3 = np.linspace(6,12,100), np.linspace(-15,2,100), np.linspace(-6,0,100)
    results = minimize(linefit, [-5, 0.8, 0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new > -8)[0]], log_M_new[np.where(log_SFR_new > -8)[0]], log_SFR_new[np.where(log_SFR_new > -8)[0]], log_Z_new[np.where(log_SFR_new > -8)[0]]), method='Nelder-Mead', tol=1e-6) # SFR > 10**-8
    log_RGW_model = results["x"][0] + results["x"][1]*log_M_new[np.where(log_SFR_new > -8)[0]] + results["x"][2]*log_SFR_new[np.where(log_SFR_new > -8)[0]] + results["x"][3]*log_Z_new[np.where(log_SFR_new > -8)[0]]
    print(results["x"][0], results["x"][1], results["x"][2], results["x"][3])
    results_2 = minimize(linefit, [-10, 0.8, -0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new <= -8)[0]], log_M_new[np.where(log_SFR_new <= -8)[0]], log_SFR_new[np.where(log_SFR_new <= -8)[0]], log_Z_new[np.where(log_SFR_new <= -8)[0]]), method='Nelder-Mead', tol=1e-6) #SFR <= 10**-8
    log_RGW_model_2 = results_2["x"][0] + results_2["x"][1]*log_M_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][2]*log_SFR_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][3]*log_Z_new[np.where(log_SFR_new <= -8)[0]]
    print(results_2["x"][0], results_2["x"][1], results_2["x"][2], results_2["x"][3])
    lin = np.linspace(-10, 0, 100)
    
    ax.scatter(log_RGW_model, log_RGW_new[np.where(log_SFR_new > -8)[0]], s=2, c='blue', alpha=0.5, label='$log(SFR)>-8$')
    ax.scatter(log_RGW_model_2, log_RGW_new[np.where(log_SFR_new <= -8)[0]], s=2, c='red', alpha=0.5, label='$log(SFR) \leq -8$')
    ax.plot(lin, lin, 'k')
    ax.set_xlabel("Predicted $R_{GW}$", fontsize=14)
    ax.set_ylabel("Actual $R_{GW}$", fontsize=14)
    ax.tick_params(direction='in')
    ax.set_ylim(-15,0)
    ax.tick_params(width=1.3)
    ax.tick_params('both',length=10, which='major')
    ax.tick_params('both',length=5, which='minor')
    for axis in ['top','left','bottom','right']:
        ax.spines[axis].set_linewidth(1.3)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(12)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(12)
    
    ax = axs[1]
    log_RGW_new = log_RGW[np.where(np.logical_and(zobs>1, zobs<=2))[0]]
    log_M_new = log_M[np.where(np.logical_and(zobs>1, zobs<=2))[0]]
    log_SFR_new = log_SFR[np.where(np.logical_and(zobs>1, zobs<=2))[0]]
    log_Z_new = log_Z[np.where(np.logical_and(zobs>1, zobs<=2))[0]]
    
    x1, x2, x3 = np.linspace(6,12,100), np.linspace(-15,2,100), np.linspace(-6,0,100)
    results = minimize(linefit, [-5, 0.8, 0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new > -8)[0]], log_M_new[np.where(log_SFR_new > -8)[0]], log_SFR_new[np.where(log_SFR_new > -8)[0]], log_Z_new[np.where(log_SFR_new > -8)[0]]), method='Nelder-Mead', tol=1e-6) # SFR > 10**-8
    log_RGW_model = results["x"][0] + results["x"][1]*log_M_new[np.where(log_SFR_new > -8)[0]] + results["x"][2]*log_SFR_new[np.where(log_SFR_new > -8)[0]] + results["x"][3]*log_Z_new[np.where(log_SFR_new > -8)[0]]
    print(results["x"][0], results["x"][1], results["x"][2], results["x"][3])
    results_2 = minimize(linefit, [-10, 0.8, -0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new <= -8)[0]], log_M_new[np.where(log_SFR_new <= -8)[0]], log_SFR_new[np.where(log_SFR_new <= -8)[0]], log_Z_new[np.where(log_SFR_new <= -8)[0]]), method='Nelder-Mead', tol=1e-6) #SFR <= 10**-8
    log_RGW_model_2 = results_2["x"][0] + results_2["x"][1]*log_M_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][2]*log_SFR_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][3]*log_Z_new[np.where(log_SFR_new <= -8)[0]]
    print(results_2["x"][0], results_2["x"][1], results_2["x"][2], results_2["x"][3])
    lin = np.linspace(-10, 0, 100)
    
    ax.scatter(log_RGW_model, log_RGW_new[np.where(log_SFR_new > -8)[0]], s=2, c='blue', alpha=0.5, label='$log(SFR)>-8$')
    ax.scatter(log_RGW_model_2, log_RGW_new[np.where(log_SFR_new <= -8)[0]], s=2, c='red', alpha=0.5, label='$log(SFR) \leq -8$')
    ax.plot(lin, lin, 'k')
    ax.set_xlabel("Predicted $R_{GW}$", fontsize=14)
    #ax.set_ylabel("Actual $R_{GW}$", fontsize=14)
    ax.tick_params(direction='in')
    ax.set_ylim(-15,0)
    ax.tick_params(width=1.3)
    ax.tick_params('both',length=10, which='major')
    ax.tick_params('both',length=5, which='minor')
    for axis in ['top','left','bottom','right']:
        ax.spines[axis].set_linewidth(1.3)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(12)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(12)
        
    ax = axs[2]
    log_RGW_new = log_RGW[np.where(np.logical_and(zobs>2, zobs<=3))[0]]
    log_M_new = log_M[np.where(np.logical_and(zobs>2, zobs<=3))[0]]
    log_SFR_new = log_SFR[np.where(np.logical_and(zobs>2, zobs<=3))[0]]
    log_Z_new = log_Z[np.where(np.logical_and(zobs>2, zobs<=3))[0]]
    
    x1, x2, x3 = np.linspace(6,12,100), np.linspace(-15,2,100), np.linspace(-6,0,100)
    results = minimize(linefit, [-5, 0.8, 0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new > -8)[0]], log_M_new[np.where(log_SFR_new > -8)[0]], log_SFR_new[np.where(log_SFR_new > -8)[0]], log_Z_new[np.where(log_SFR_new > -8)[0]]), method='Nelder-Mead', tol=1e-6) # SFR > 10**-8
    log_RGW_model = results["x"][0] + results["x"][1]*log_M_new[np.where(log_SFR_new > -8)[0]] + results["x"][2]*log_SFR_new[np.where(log_SFR_new > -8)[0]] + results["x"][3]*log_Z_new[np.where(log_SFR_new > -8)[0]]
    print(results["x"][0], results["x"][1], results["x"][2], results["x"][3])
    results_2 = minimize(linefit, [-10, 0.8, -0.5, 0.01], args=(log_RGW_new[np.where(log_SFR_new <= -8)[0]], log_M_new[np.where(log_SFR_new <= -8)[0]], log_SFR_new[np.where(log_SFR_new <= -8)[0]], log_Z_new[np.where(log_SFR_new <= -8)[0]]), method='Nelder-Mead', tol=1e-6) #SFR <= 10**-8
    log_RGW_model_2 = results_2["x"][0] + results_2["x"][1]*log_M_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][2]*log_SFR_new[np.where(log_SFR_new <= -8)[0]] + results_2["x"][3]*log_Z_new[np.where(log_SFR_new <= -8)[0]]
    print(results_2["x"][0], results_2["x"][1], results_2["x"][2], results_2["x"][3])
    lin = np.linspace(-10, 0, 100)
    
    ax.scatter(log_RGW_model, log_RGW_new[np.where(log_SFR_new > -8)[0]], s=2, c='blue', alpha=0.5, label='$log(SFR)>-8$')
    ax.scatter(log_RGW_model_2, log_RGW_new[np.where(log_SFR_new <= -8)[0]], s=2, c='red', alpha=0.5, label='$log(SFR) \leq -8$')
    ax.plot(lin, lin, 'k')
    ax.set_xlabel("Predicted $R_{GW}$", fontsize=14)
    #ax.set_ylabel("Actual $R_{GW}$", fontsize=14)
    ax.tick_params(direction='in')
    ax.set_ylim(-15,0)
    ax.tick_params(width=1.3)
    ax.tick_params('both',length=10, which='major')
    ax.tick_params('both',length=5, which='minor')
    for axis in ['top','left','bottom','right']:
        ax.spines[axis].set_linewidth(1.3)
    for tick in ax.xaxis.get_ticklabels():
        tick.set_fontsize(12)
    for tick in ax.yaxis.get_ticklabels():
        tick.set_fontsize(12)
    ax.legend()
    plt.show()
    
    
    plt.figure()
    
    cax = plt.hexbin(log_RGW_model, log_RGW[index][np.where(log_SFR[index] > -8)[0]], C = np.abs(log_RGW_model - log_RGW[index][np.where(log_SFR[index] > -8)[0]]), gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func)
    cbar = plt.colorbar(cax)
    cbar.set_label("Residual", fontsize=12.5, rotation=270, labelpad = 18)
    plt.show()
    
    #X, Y, Z = np.meshgrid(x1, x2, x3)
    index_2 = np.where(log_RGW>= -10)[0]
    
    fig = plt.figure()
    axis = fig.add_subplot(111, projection='3d')
    #axis.plot(x1, x2, results["x"][0] + results["x"][1]*x1 + results["x"][2]*x2 + results["x"][3]*x3, color='red', zorder=10)
    #axis.plot(x1, x2, results_2["x"][0] + results_2["x"][1]*x1 + results_2["x"][2]*x2 + results_2["x"][3]*x3, color='red', zorder=10)
    #plot3d = axis.scatter(log_M[index_2], log_SFR[index_2], log_RGW[index_2], c = log_Z[index_2], zorder=1)
    plot3d = axis.scatter(log_M, log_SFR, log_RGW, c = log_Z, zorder=1)
    #axis.plot_surface(X, Y, results["x"][0] + results["x"][1]*X + results["x"][2]*Y + results["x"][3]*Z)
    #axis.plot(log_M, log_SFR, 'k+', zdir='z', zs=-7)
    #axis.scatter(np.log10(M_stellar[0:len(R_GW['y'])]), np.log10(SFR_obs[0:len(R_GW['y'])]), np.zeros(len(R_GW['y'])), c = Z_obs[0:len(R_GW['y'])])
    #ax = plt.axes(projection="3d")
    #ax.plot3D(M_stellar[0:len(R_GW['y'])], SFR_obs[0:len(R_GW['y'])], R_GW['y'], 'gray')
    cbar=plt.colorbar(plot3d, pad = 0.2)
    #plt.ticklabel_format(axis='x', style='sci')
    cbar.set_label("$\log(Z)$", labelpad= 15, fontsize=14)
    axis.set_xlabel("$\log(M_* [M_{\odot}])$", labelpad=15, fontsize=14)
    axis.set_ylabel("$\log(SFR [M_{\odot} /yr])$", labelpad=15, fontsize=14)
    axis.set_zlabel("$\log(R_{GW, obs} [yr^{-1}])$", labelpad=15, fontsize=14)
    #plt.savefig('/Users/s4431433/Documents/codes/Paper plots subvolume 1/R_GW_3d_linfit.png')
    plt.show()
    
    
    fig = plt.figure()
    cmap = plt.get_cmap('gray_r')
    counts,xbins,ybins = np.histogram2d(ph_data['y1'] - ph_data['y4'], ph_data['y0'] - ph_data['y5'],bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for l in range(len(mylevels)):
        ival = l*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
            
                    
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
                    
    cax = plt.hexbin(ph_data['y1'] - ph_data['y4'], ph_data['y0'] - ph_data['y5'], C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-7.0, vmax=np.log10(0.3463695029598574))
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%d}$"))
    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", rotation=270, labelpad = 15)
    plt.vlines(-0.5, 0, 5.2, 'r', '--')
    plt.ylim(0, 5.2)
    plt.hlines(3, -1.5, 1.5, 'r', '--')
    plt.xlim(-1.5, 1.5)
    plt.fill_between([-1.5,-0.5], 3, 5.2, color='grey', alpha =0.25)
    plt.xlabel('J-g')
    plt.ylabel('u-r')
    plt.savefig('/Users/Liana/Documents/PYTHON/New_plots/color_color_2.png')
    plt.show()

    index = np.where(np.logical_and(ph_data['y1'] - ph_data['y4'] < -0.5, ph_data['y0'] - ph_data['y5'] > 3.0, ph_data['y0'] - ph_data['y5'] <= 5.0))[0]
    print(len(index))
    R_sum_cutoff = np.sum(data['y4'][index])
    print(R_sum_cutoff/len(index))
    print(np.sum(data['y4'])/47316)

    #/(np.log(10)*bin_means[index_count])
    index = np.where( np.logical_and(log_M > 6.002, log_M < 6.004))
    plt.scatter(log_SFR[index], np.log10(data['y4'])[index], color = 'b')
    index = np.where( np.logical_and(log_M > 6.004, log_M < 6.006))
    plt.scatter(log_SFR[index], np.log10(data['y4'])[index], color = 'g')
    
    
    Chabrier_data = np.load('Chabrier_data.npz') # STANDARD
    Miller_data = np.load('Miller_data.npz')
    Kroupa_data = np.load('Kroupa_data.npz')
    Case2_data = np.load('Chabrier_case_2.npz')
    Case3_data = np.load('Chabrier_case_3.npz')
    qcut3 = np.load('Chabrier_q_3.npz')
    noqcut = np.load('Chabrier_q_0.npz')
    beta73 = np.load('Chabrier_b_7_3.npz')
    beta133 = np.load('Chabrier_b_1_3.npz')
    tmin93 = np.load('Chabrier_tmin_9_3.npz')
    tmin54 = np.load('Chabrier_tmin_8_6.npz')
    
    fig, axs = plt.subplots(nrows=1, ncols=2)
    ax = axs[0]
    R_plot = ax.scatter(lbt[0:len(all_MFH[0])], tmin93['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])

    ax.scatter(lbt[0:len(all_MFH[0])], tmin54['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])
    #cb = plt.colorbar(R_plot)
    #cb.set_label('Z', rotation=270, labelpad=10)
    ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], '--', color='grey', label="$t_{min}=10^{7.5}yr$")
    ax.plot(lbt[0:len(all_MFH[0])], tmin54['y4'][0:len(all_MFH[0])], '--', color='black', label = "$t_{min}=10^{8.6}yr$")
    ax.plot(lbt[0:len(all_MFH[0])], tmin93['y4'][0:len(all_MFH[0])], '--', color='red', label = "$t_{min}=10^{9.3}yr$")
    
    ax.set_ylim(0, None)
    ax.legend(loc="lower left")
    ax.set_xlabel("Lookback time (Gyr)", fontsize=12.5)
    ax.set_ylabel("$R_\mathrm{GW}$ ($yr^{-1}$)", fontsize=12.5)
    
    ax2 = ax.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])
    ax2.set_xlabel("Redshift", fontsize=12.5)
    
    ax = axs[1]
    R_plot = ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y3'][0:len(all_MFH[0])]/2, s=15, c = all_MFH[0])
    cb = plt.colorbar(R_plot, ax=ax)
    cb.set_label('Z', rotation=270, fontsize = 12.5, labelpad=10)
    ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y3'][0:len(all_MFH[0])]/2, '--', color='grey', label="$t_{delay}=0yr$")
    ax.set_ylim(0, None)
    ax.legend(loc="upper left")
    ax.set_xlabel("Lookback time (Gyr)", fontsize=12.5)
    ax.set_ylabel("$R_\mathrm{GW}$ ($yr^{-1}$)", fontsize=12.5)
    ax2 = ax.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])
    ax2.set_xlabel("Redshift", fontsize=12.5)
    #plt.savefig("/Users/Liana/Documents/PYTHON/New_plots/RGW_t_min_2.pdf")
    plt.show()
    

    fig = plt.figure()
    ax = fig.add_subplot(111)
    R_plot = ax.scatter(lbt[0:len(all_MFH[0])], noqcut['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], qcut3['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], s=15, c = all_MFH[0])
    cb = plt.colorbar(R_plot)
    cb.set_label('Z', rotation=270, labelpad=10)
    ax.plot(lbt[0:len(all_MFH[0])], qcut3['y4'][0:len(all_MFH[0])], '--', color='black', label = "$q_{cut}=0.3$")
    ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], '--', color='grey', label="$q_{cut}=0.5$")
    ax.plot(lbt[0:len(all_MFH[0])], noqcut['y4'][0:len(all_MFH[0])], '--', color='red', label = " $q_{cut}=0.8$")

    ax.legend(loc="lower left")
    ax.set_xlabel("Lookback time (Gyr)", fontsize=12.5)
    ax.set_ylabel("$R_{GW}$ ($yr^{-1}$)", fontsize=12.5)
    
    ax2 = ax.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])
    ax2.set_xlabel("Redshift", fontsize=12.5)
    
    plt.savefig("/Users/Liana/Documents/PYTHON/New_plots/RGW_q_cutoff.pdf")
    plt.show()
    
    
    fig, axs = plt.subplots(nrows=2, figsize=(7, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax = axs[0]
    
    #R_plot = ax.scatter(lbt[0:len(all_MFH[0])], noqcut['y4'][0:len(all_MFH[0])], s=10, c = all_MFH[0])
    #ax.scatter(lbt[0:len(all_MFH[0])], qcut3['y4'][0:len(all_MFH[0])], s=10, c = all_MFH[0])
    #ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], s=10, c = all_MFH[0])
    
    R_plot = ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y1'][0:len(all_MFH[0])], s=13, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Case2_data['y1'][0:len(all_MFH[0])], s=13, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Case3_data['y1'][0:len(all_MFH[0])], s=13, c = all_MFH[0])
    ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y1'][0:len(all_MFH[0])], '--', color="grey", label="Case i")
    ax.plot(lbt[0:len(all_MFH[0])], Case2_data['y1'][0:len(all_MFH[0])], '--', color="black", label="Case ii")
    ax.plot(lbt[0:len(all_MFH[0])], Case3_data['y1'][0:len(all_MFH[0])], '--', color="red", label="Case iii")
    
    #ax.plot(lbt[0:len(all_MFH[0])], qcut3['y4'][0:len(all_MFH[0])], '--', color='black', label = "$q_{cut}=0.3$")
    #ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y4'][0:len(all_MFH[0])], '--', color='grey', label="$q_{cut}=0.5$")
    #ax.plot(lbt[0:len(all_MFH[0])], noqcut['y4'][0:len(all_MFH[0])], '--', color='red', label = " $q_{cut}=0.8$")
    
    ax2 = ax.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])

    ax2.set_xlabel("Redshift", fontsize=13)
    
    #ax.set_yscale('log')
    #ax.set_ylim(0, 0.025)
    cb = fig.colorbar(R_plot, ax=ax)
    cb.set_label('Z', rotation=270, fontsize=13, labelpad=10)
    ax.legend(loc="upper left")
    #ax.set_ylabel("BHFR ($M_\odot/yr$ per galaxy)", fontsize=12)
    ax.set_ylabel(r"$R_\mathrm{BH\ mass}$ ($M_\odot/yr$)", fontsize=13)
    ax.set_xlabel("Lookback time (Gyr)", fontsize=13)
    
    ax = axs[1]
    
    #ax.plot(lbt[0:len(all_MFH[0])], abs(Chabrier_data['y4'][0:len(all_MFH[0])]-qcut3['y4'][0:len(all_MFH[0])]), color="black", label = "|$R_{GW, >0.5} - R_{GW, >0.3}$|")
    #ax.plot(lbt[0:len(all_MFH[0])], abs(Chabrier_data['y4'][0:len(all_MFH[0])]-noqcut['y4'][0:len(all_MFH[0])]), color="red", label = "|$R_{GW, >0.5} - R_{GW, >0.8}$|")
    
    ax.plot(lbt[0:len(all_MFH[0])], Case2_data['y1'][0:len(all_MFH[0])]/Chabrier_data['y1'][0:len(all_MFH[0])], color="black", label="Case ii/Case i")
    ax.plot(lbt[0:len(all_MFH[0])], Case3_data['y1'][0:len(all_MFH[0])] /Chabrier_data['y1'][0:len(all_MFH[0])], color="red", label="Case iii/Case i")
    
    ax.set_ylabel(r"$R_\mathrm{BH\ mass}$ Ratio", fontsize=13)
    #ax.set_yscale("log")
    #ax.set_ylabel("Error", fontsize=13)
    ax.set_xlabel("Lookback time (Gyr)", fontsize=13)
    ax.legend(loc="upper right")
    #plt.savefig("/Users/Liana/Documents/PYTHON/New_plots/RGW_q_cutoff.png")
    plt.savefig("/Users/Liana/Documents/PYTHON/New_plots/Mrem_comparison_BHFR.pdf")
    plt.show()
    
    
    #fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
    fig, axs = plt.subplots(nrows=2, figsize=(7, 10), gridspec_kw={'height_ratios': [3, 1]})
    ax = axs[0]
    R_plot = ax.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y3'][0:len(all_MFH[0])], s = 13, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Miller_data['y3'][0:len(all_MFH[0])], s = 13, c = all_MFH[0])
    ax.scatter(lbt[0:len(all_MFH[0])], Kroupa_data['y3'][0:len(all_MFH[0])], s = 13, c = all_MFH[0])
    ax.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y3'][0:len(all_MFH[0])], '--', color="grey", label = "Chabrier")
    ax.plot(lbt[0:len(all_MFH[0])], Miller_data['y3'][0:len(all_MFH[0])], '--', color="black", label = "Miller-Scalo")
    ax.plot(lbt[0:len(all_MFH[0])], Kroupa_data['y3'][0:len(all_MFH[0])], '--', color="red", label = "Kroupa")
    #ax.set_ylim(0, 0.03)
    ax2 = ax.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])
    ax2.set_xlabel("Redshift", fontsize=13)
    ax.legend(loc="upper left")
    cb = fig.colorbar(R_plot, ax=ax)
    cb.set_label('Z', rotation=270, fontsize=13, labelpad=10)
    ax.set_xlabel("Lookback time (Gyr)", fontsize=13)
    ax.set_ylabel("$R_\mathrm{birth}$ ($yr^{-1}$)", fontsize=13)
    
    ax = axs[1]
    ax.plot(lbt[0:len(all_MFH[0])], Miller_data['y3'][0:len(all_MFH[0])]/Chabrier_data['y3'][0:len(all_MFH[0])], color="black", label = "Miller-Scalo/Chabrier")
    ax.plot(lbt[0:len(all_MFH[0])], Kroupa_data['y3'][0:len(all_MFH[0])]/Chabrier_data['y3'][0:len(all_MFH[0])], color="red", label = "Kroupa/Chabrier")
    ax.legend(loc="lower right")
    ax.set_ylabel("$R_\mathrm{birth}$ Ratio", fontsize=13)
    #ax.set_ylabel("Error", fontsize=12)
    #ax.set_ylabel("$\u0394 R_{GW}$")
    ax.set_ylim(0.8, 1.4)
    ax.set_xlabel("Lookback time (Gyr)", fontsize=13)
    plt.savefig("/Users/Liana/Documents/PYTHON/New_plots/IMF_comparison_Rbirth.pdf.pdf")
    plt.show()
    
    fig, axs = plt.subplots(nrows=1, ncols=3)
    
    ax = axs[0]
    index = np.where(np.logical_and(data['y5']>=0, data['y5']<1))[0]
    cmap = plt.get_cmap('gray_r')
    counts,xbins,ybins = np.histogram2d(log_M[index], log_sSFR[index],bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
        #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = ax.hexbin(log_M[index], log_sSFR[index], C = data['y4'][index], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_sum_func, vmin=-6, vmax=0)
    #plt.vlines(10.2, -14, 2, 'r', '--')
    #plt.hlines(-1.5, 7, 12, 'r', '--')
    #plt.xlim(7, 12)
    #plt.ylim(-14, 2)
    #plt.fill_between([10.2,12], -1.5, 2, color='grey', alpha =0.25)
    ax.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    #ax.set_xlabel("$\log(Z)$", fontsize=12.5)
    #ax.set_ylabel("$\log(SFR [M_{\odot}/yr])$", fontsize=12.5)
    ax.set_xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    #text_kwargs = dict(ha='center', va='center', fontsize=16, color='k')
    #ax.text(11.7, -17.75, '$z<0.3$', **text_kwargs)
    ax.set_ylabel("$\log(sSFR [yr^{-1}])$", fontsize=12.5)
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #cb=fig.colorbar(cax, cax=cbar_ax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    #cbar = ax.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    #cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", fontsize=12.5, rotation=270, labelpad = 18)
    #plt.savefig('/Users/s4431433/Documents/codes/Paper plots subvolume 1/hexbin_SFRvsM_mean.png')
    
    ax = axs[1]
    cmap = plt.get_cmap('gray_r')
    index = np.where(np.logical_and(data['y5']>=1, data['y5']<2))[0]
    counts,xbins,ybins = np.histogram2d(log_M[index], log_sSFR[index],bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
        #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = ax.hexbin(log_M[index], log_sSFR[index], C = data['y4'][index], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_sum_func, vmin=-6, vmax=0)
    
    ax.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.xlabel("$\log(Z)$", fontsize=12.5)
    #ax.set_ylabel("$\log(SFR [M_{\odot}/yr])$", fontsize=12.5)
    ax.set_xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    #ax.set_xlabel("$\log(Z)$", fontsize=12.5)
    #text_kwargs = dict(ha='center', va='center', fontsize=16, color='k')
    #ax.text(11.4, -17.75, '$0.3 \leq z<1$', **text_kwargs)
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    #fig.subplots_adjust(right=0.8)
    #cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    #cb=fig.colorbar(cax, cax=cbar_ax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    
    ax = axs[2]
    cmap = plt.get_cmap('gray_r')
    index = np.where(np.logical_and(data['y5']>=2, data['y5']<3))[0]
    counts,xbins,ybins = np.histogram2d(log_M[index], log_sSFR[index],bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
        #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = ax.hexbin(log_M[index], log_sSFR[index], C = data['y4'][index], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_sum_func, vmin=-6, vmax=0)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    #plt.xlabel("$\log(Z)$", fontsize=12.5)
    #ax.set_ylabel("$\log(SFR [M_{\odot}/yr])$", fontsize=12.5)
    ax.set_xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    #text_kwargs = dict(ha='center', va='center', fontsize=16, color='k')
    #ax.text(11.2, -17.4, '$1 \leq z<2$', **text_kwargs)
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    #cbar = ax.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cb=fig.colorbar(cax, cax=cbar_ax, format=mtick.FormatStrFormatter(r"$10^{%.1f}$"))
    cb.set_label("Total $R_{GW, obs} [yr^{-1}]$", fontsize=12.5, rotation=270, labelpad = 18)
    plt.show()
    
    fig, axs = plt.subplots(nrows=1, ncols=3)
    ax = axs[0]
    len_0, bin_edges_0 = np.histogram(ph_data['y1'][np.where(ph_data['y1'] != -999)[0]], bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(ph_data['y1'][np.where(ph_data['y1'] != -999)[0]], bins=50, weights = data['y4'][np.where(ph_data['y1'] != -999)[0]])

    len_1, bin_edges_1 = np.histogram(ph_data_2['y1'][np.where(ph_data_2['y1'] != -999)[0]], bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(ph_data_2['y1'][np.where(ph_data_2['y1'] != -999)[0]], bins=50, weights = data_2['y4'][np.where(ph_data_2['y1'] != -999)[0]])
    
    len_2, bin_edges_2 = np.histogram(ph_data_3['y1'][np.where(ph_data_3['y1'] != -999)[0]], bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(ph_data_3['y1'][np.where(ph_data_3['y1'] != -999)[0]], bins=50, weights = data_3['y4'][np.where(ph_data_3['y1'] != -999)[0]])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("J")
    ax.set_ylabel("$N_{GW}/N_{gal}$")
    ax.set_yscale("log")
    plt.legend()
    
    ax = axs[1]
    len_0, bin_edges_0 = np.histogram(ph_data['y3'][np.where(ph_data['y3'] != -999)[0]], bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(ph_data['y3'][np.where(ph_data['y3'] != -999)[0]], bins=50, weights = data['y4'][np.where(ph_data['y3'] != -999)[0]])

    len_1, bin_edges_1 = np.histogram(ph_data_2['y3'][np.where(ph_data_2['y3'] != -999)[0]], bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(ph_data_2['y3'][np.where(ph_data_2['y3'] != -999)[0]], bins=50, weights = data_2['y4'][np.where(ph_data_2['y3'] != -999)[0]])
    
    len_2, bin_edges_2 = np.histogram(ph_data_3['y3'][np.where(ph_data_3['y3'] != -999)[0]], bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(ph_data_3['y3'][np.where(ph_data_3['y3'] != -999)[0]], bins=50, weights = data_3['y4'][np.where(ph_data_3['y3'] != -999)[0]])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("K")
    ax.set_yscale("log")
    
    ax = axs[2]
    len_0, bin_edges_0 = np.histogram(ph_data['y5'][np.where(ph_data['y5'] != -999)[0]], bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(ph_data['y5'][np.where(ph_data['y5'] != -999)[0]], bins=50, weights = data['y4'][np.where(ph_data['y5'] != -999)[0]])

    len_1, bin_edges_1 = np.histogram(ph_data_2['y5'][np.where(ph_data_2['y5'] != -999)[0]], bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(ph_data_2['y5'][np.where(ph_data_2['y5'] != -999)[0]], bins=50, weights = data_2['y4'][np.where(ph_data_2['y5'] != -999)[0]])
    
    len_2, bin_edges_2 = np.histogram(ph_data_3['y5'][np.where(ph_data_3['y5'] != -999)[0]], bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(ph_data_3['y5'][np.where(ph_data_3['y5'] != -999)[0]], bins=50, weights = data_3['y4'][np.where(ph_data_3['y5'] != -999)[0]])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("r")
    #ax.set_ylabel("$N_{GW}/N_{gal}$")
    ax.set_yscale("log")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.figure()
    #plt.hist(ph_data['y0'][np.where(ph_data['y0'] != -999)[0]], bins = 30, weights= hist_2/hist)
    plt.show()
    
    fig, axs = plt.subplots(nrows=1, ncols=3)
    ax = axs[0]
    len_0, bin_edges_0 = np.histogram(log_M, bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(log_M, bins=50, weights = data['y4'])

    len_1, bin_edges_1 = np.histogram(log_M_2, bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(log_M_2, bins=50, weights = data_2['y4'])
    
    len_2, bin_edges_2 = np.histogram(log_M_3, bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(log_M_3, bins=50, weights = data_3['y4'])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("$\log(M_* [M_{\odot}])$")
    ax.set_ylabel("$N_{GW}/N_{gal}$")
    ax.set_yscale("log")
    
    ax = axs[1]
    len_0, bin_edges_0 = np.histogram(log_SFR, bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(log_SFR, bins=50, weights = data['y4'])

    len_1, bin_edges_1 = np.histogram(log_SFR_2, bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(log_SFR_2, bins=50, weights = data_2['y4'])
    
    len_2, bin_edges_2 = np.histogram(log_SFR_3, bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(log_SFR_3, bins=50, weights = data_3['y4'])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("$\log(SFR [M_{\odot}/yr])$")
    #ax.set_ylabel("$N_{GW}/N_{gal}$")
    ax.set_yscale("log")
    
    ax = axs[2]
    len_0, bin_edges_0 = np.histogram(log_Z, bins=50)
    bin_mid = 0.5*(bin_edges_0[:-1] + bin_edges_0[1:])
    RGW_tot_0, bin_edges = np.histogram(log_Z, bins=50, weights = data['y4'])

    len_1, bin_edges_1 = np.histogram(log_Z_2, bins=50)
    bin_mid_1 = 0.5*(bin_edges_1[:-1] + bin_edges_1[1:])
    RGW_tot_1, bin_edges = np.histogram(log_Z_2, bins=50, weights = data_2['y4'])
    
    len_2, bin_edges_2 = np.histogram(log_Z_3, bins=50)
    bin_mid_2 = 0.5*(bin_edges_2[:-1] + bin_edges_2[1:])
    RGW_tot_2, bin_edges = np.histogram(log_Z_3, bins=50, weights = data_3['y4'])
    
    ax.step(bin_mid, RGW_tot_0/len_0, label="$z<0.3$")
    ax.step(bin_mid_1, RGW_tot_1/len_1, label="$0.3 \leq z < 1$")
    ax.step(bin_mid_2, RGW_tot_2/len_2, label="$1 \leq z < 2$")
    ax.set_xlabel("$\log(Z)$")
    #ax.set_ylabel("$N_{GW}/N_{gal}$")
    ax.set_yscale("log")
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    #plt.figure()
    #plt.hist(ph_data['y0'][np.where(ph_data['y0'] != -999)[0]], bins = 30, weights= hist_2/hist)
    plt.show()
    '''
    ###############################################################################################

    N = 30
    RGW_all = []
    RGW_all.extend(data['y4'])
    #RGW_all.extend(data_2['y4'])
    #RGW_all.extend(data_3['y4'])
    #RGW_all.extend(data_4['y4'])
    RGW_all = np.array(RGW_all)
    SFR_all = []
    SFR_all.extend(data['y2']) # GIVEN IN M/GYR
    #SFR_all.extend(data_2['y2'])
    #SFR_all.extend(data_3['y2'])
    #SFR_all.extend(data_4['y2'])
    SFR_all = np.array(SFR_all)
    log_sSFR_all = []
    log_sSFR_all.extend(log_sSFR)
    #log_sSFR_all.extend(log_sSFR_2)
    #log_sSFR_all.extend(log_sSFR_3)
    #log_sSFR_all.extend(log_sSFR_4)
    log_sSFR_all = np.array(log_sSFR_all)
    z_all = []
    z_all.extend(data['y5'])
    #z_all.extend(data_2['y5'])
    #z_all.extend(data_3['y5'])
    #z_all.extend(data_4['y5'])
    z_all = np.array(z_all)
    subvol_list = [0, 1, 2, 7, 8, 9, 10, 12, 13, 15]
    RGW_sum = np.zeros((len(subvol_list), N))
    RGW_err = np.zeros(N)
    RGW_ave = np.zeros(N)
    RGW_tot, bin_edges = np.histogram(z_all, bins = N, weights = RGW_all)
    print(len(RGW_tot))
    RGW_tot_1, _ = np.histogram(z_all[np.where(log_sSFR_all < -10)[0]], bins = N, weights = RGW_all[np.where(log_sSFR_all < -10)[0]])
    RGW_tot_2, _ = np.histogram(z_all[np.where(log_sSFR_all >= -10)[0]], bins = bin_edges, weights = RGW_all[np.where(log_sSFR_all >= -10)[0]])
    RGW_count, _ = np.histogram(z_all, bins = bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    dist = cosmo.comoving_distance(bin_mid)
    radius = cosmo.comoving_distance(bin_edges)
    volume = Normalisation_factor*(1.6857656436069093/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    '''
    RGW_vol_tot = np.zeros((1000, N))
    RGW_vol_tot_2 = np.zeros((1000, N))
    for i in range(len(subvol_list)):
        RGW = np.load(str('properties_subvol_%d.npz' % subvol_list[i]))['y4']
        RGW_tot_new, _ = np.histogram(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6'], bins = bin_edges, weights = RGW)
        RGW_sum[i,:] = RGW_tot_new/volume
        #RGW_sum[i,:] = RGW_tot_new
    for i in range(10):
        for j in range(N):
            boot = resample(RGW_sum[:,j], replace=True, n_samples=10, random_state=1)
            RGW_vol_tot[i,j] = np.sum(boot)
    for i in range(N):
        boot = resample(RGW_vol_tot[:,i], replace=True, n_samples=100, random_state=1)
        RGW_err[i] = np.std(boot)/np.sqrt(len(boot))
        RGW_ave[i] = np.mean(boot)
    '''
    
    #plt.errorbar(dist.value, f_eff*RGW_tot/volume.value, yerr=f_eff*RGW_err, capsize=2, fmt="none", ecolor="k")
    RGW_all = []
    RGW_all.extend(data['y4'])
    RGW_all = np.array(RGW_all)
    z_all = []
    z_all.extend(data['y5'])
    z_all = np.array(z_all)
    u_mag_all = []
    u_mag_all.extend(ph_data['y0'])
    #u_mag_all.extend(ph_data_2['y0'])
    #u_mag_all.extend(ph_data_3['y0'])
    #u_mag_all.extend(ph_data_4['y0'])
    u_mag_all = np.array(u_mag_all)
    J_mag_all = []
    J_mag_all.extend(ph_data['y1'])
    #J_mag_all.extend(ph_data_2['y1'])
    #J_mag_all.extend(ph_data_3['y1'])
    #J_mag_all.extend(ph_data_4['y1'])
    J_mag_all = np.array(J_mag_all)
    K_mag_all = []
    K_mag_all.extend(ph_data['y3'])
    #K_mag_all.extend(ph_data_2['y3'])
    #K_mag_all.extend(ph_data_3['y3'])
    #K_mag_all.extend(ph_data_4['y3'])
    K_mag_all = np.array(K_mag_all)
    g_mag_all = []
    g_mag_all.extend(ph_data['y4'])
    #g_mag_all.extend(ph_data_2['y4'])
    #g_mag_all.extend(ph_data_3['y4'])
    #g_mag_all.extend(ph_data_4['y4'])
    g_mag_all = np.array(g_mag_all)
    r_mag_all = []
    r_mag_all.extend(ph_data['y5'])
    #r_mag_all.extend(ph_data_2['y5'])
    #r_mag_all.extend(ph_data_3['y5'])
    #r_mag_all.extend(ph_data_4['y5'])
    r_mag_all = np.array(r_mag_all)
    i_mag_all = []
    i_mag_all.extend(ph_data['y6'])
    #i_mag_all.extend(ph_data_2['y6'])
    #i_mag_all.extend(ph_data_3['y6'])
    #i_mag_all.extend(ph_data_4['y6'])
    i_mag_all = np.array(i_mag_all)
    z_mag_all = []
    z_mag_all.extend(ph_data['y7'])
    #z_mag_all.extend(ph_data_2['y7'])
    #z_mag_all.extend(ph_data_3['y7'])
    #z_mag_all.extend(ph_data_4['y7'])
    z_mag_all = np.array(z_mag_all)
    y_mag_all = []
    y_mag_all.extend(ph_data['y9'])
    #y_mag_all.extend(ph_data_2['y9'])
    #y_mag_all.extend(ph_data_3['y9'])
    #y_mag_all.extend(ph_data_4['y9'])
    y_mag_all = np.array(y_mag_all)
    
    SFR_tot, _ = np.histogram(z_all, bins = bin_edges, weights = SFR_all)
    RGW_num, _ = np.histogram(z_all, bins = bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    data = h5py.File('/Users/s4431433/Documents/codes/group/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/199/0/star_formation_histories.hdf5', 'r')
    lbt = np.array(data.get('lbt_mean'))
    redshift = np.array(data.get('redshifts'))
    SFR_cosmic = 0.63*0.015*(1+redshift)**2.7/(1+((1+redshift)/2.9)**5.6) # CONVERT TO CHABRIER IMF FROM SALPETER
    sfr_vals = np.array([-1.95, -1.82, -1.90, -1.77, -1.75, -1.79, -1.73, -1.56, -1.42, -1.29, -1.31, -1.27, -1.17, -1.30, -1.29, -1.28, -1.33, -1.42, -1.45])
    z_lower = np.array([0.02, 0.06, 0.14, 0.2, 0.28, 0.36, 0.45, 0.56, 0.68, 0.82, 1.0, 1.2, 1.45, 1.75, 2.2, 2.6, 3.25, 3.75, 4.25])
    z_upper = np.array([0.08, 0.14, 0.2, 0.28, 0.36, 0.45, 0.56, 0.68, 0.82, 1.0, 1.2, 1.45, 1.75, 2.2, 2.6, 3.25, 3.75, 4.25, 5])
    z_mid = 0.5*(z_lower+z_upper)
    dist = cosmo.comoving_distance(bin_mid)
    #print(dist)
    #print(bin_mid)
    #print(RGW_tot[np.argsort(RGW_tot)])
    #print(RGW_tot*10**9)
    #print(RGW_tot*10**9/volume)
    #print(SFR_tot)
    dist = cosmo.comoving_distance(bin_mid)
    dist = dist*10**-3 #Gpc
    cosmo = FlatLambdaCDM(H0=100, Om0=0.27)
    r_mag_driver = np.array([-23.25, -22.75, -22.25, -21.75, -21.25, -20.75, -20.25, -19.75, -19.25, -18.75, -18.25, -17.75, -17.25, -16.75, -16.25, -15.75, -15.25, -14.75, -14.25, -13.75])
    lf_driver = np.array([7.5587418E-06, 1.51174836E-05, 9.82636629E-05, 0.00051777414, 0.00142481795, 0.00229406729, 0.00379074016, 0.00481876126, 0.00521182828, 0.00599418255, 0.00671986025, 0.00718391687, 0.00787164923, 0.00935216155, 0.0105548939, 0.0143988617, 0.0150681427, 0.0140159791, 0.00799279194, 0.00878074206])
    lf_error = np.array([5.34483752E-06, 7.5587418E-06, 1.92710886E-05, 4.42364326E-05, 7.33818524E-05, 9.31133181E-05, 0.000119694327, 0.000134952308, 0.000140348566, 0.000150514519, 0.000163317149, 0.000214469153, 0.000311641576, 0.000461870339, 0.000688525324, 0.0011528316, 0.00165394356, 0.00227369205, 0.00302099157, 0.00620892225])
    z_list = [0, 1, 2, 3]
    lns = ['solid', 'dotted', 'dashed']
    colours = ['b', 'r', 'g']
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    data_1 = np.load('all_subvols.npz')
    r_mag_all = data_1['y2']
    z_all = data_1['y0']
    print(cosmo.comoving_distance(0))
    hf = h5py.File('/Users/s4431433/Downloads/all_r_mag.hdf5', 'r')
    for i in range(4):
        r_mag_all = np.array(hf.get(str('z=%d' %i)))
        #index = np.where(np.logical_and(np.logical_and(z_all>=z_list[i], z_all<z_list[i]+0.05), r_mag_all != -999))[0]
        #print(len(index))
        #N = math.ceil((np.amax(r_mag_all[index])-np.amin(r_mag_all[index]))/0.5)
        N = math.ceil((np.amax(r_mag_all)-np.amin(r_mag_all))/0.5)
        #r_mag_dist, r_bin_edges = np.histogram(r_mag_all[index]-5*np.log10(0.678), bins=N)
        r_mag_dist, r_bin_edges = np.histogram(r_mag_all-5*np.log10(0.678), bins=N)
        #r_mag_dist_bulge_d, _ = np.histogram(ph_data['y11'][index], bins=r_bin_edges)
        #r_mag_dist_bulge_m, _ = np.histogram(ph_data['y12'][index], bins=r_bin_edges)
        #r_mag_dist_disk, _ = np.histogram(ph_data['y13'][index], bins=r_bin_edges)
        r_bin_mid = 0.5*(r_bin_edges[:-1] + r_bin_edges[1:])
        #radius = cosmo.comoving_distance(z_list[i]+0.05)**3-cosmo.comoving_distance(z_list[i])**3
        #print(radius)
        #r_vol = (1.6857656436069093*64/41252.96) * (4*np.pi*(radius)/3)
        #print(r_vol)
        #ax1.plot(r_bin_mid, np.log10(r_mag_dist/((r_bin_edges[1:]-r_bin_edges[:-1])*r_vol.value)), label=str(z_list[i]) + "<=" + "z" + "<" + str(z_list[i+1]))
        #ax1.plot(r_bin_mid, np.log10(r_mag_dist/r_vol.value), label="z="+str(z_list[i]))
        ax1.plot(r_bin_mid, np.log10(r_mag_dist/210**3), label="z="+str(z_list[i]))
        #plt.plot(r_bin_mid, np.log10(r_mag_dist_bulge_d/((r_bin_edges[1:]-r_bin_edges[:-1])*r_vol.value)), color='r', linestyle=lns[i], label="bulge disk instabilities")
        #plt.plot(r_bin_mid, np.log10(r_mag_dist_bulge_m/((r_bin_edges[1:]-r_bin_edges[:-1])*r_vol.value)), color='b', linestyle=lns[i], label="bulge mergers")
        #plt.plot(r_bin_mid, np.log10(r_mag_dist_disk/((r_bin_edges[1:]-r_bin_edges[:-1])*r_vol.value)), color='g', linestyle=lns[i], label="disk")
        
        if i == 0:
            line = plt.plot(0,0, color='k', linestyle='solid', label="$0 \leq z < 1$")
            line1 = plt.plot(0,0, color='k', linestyle='dotted', label="$1 \leq z < 2$")
            line2 = plt.plot(0,0, color='k', linestyle='dashed', label="$2 \leq z < 3$")
            plt.legend()
            line = line.pop(0)
            line.remove()
            line1 = line1.pop(0)
            line1.remove()
            line2 = line2.pop(0)
            line2.remove()
        
    
    ax1.errorbar(r_mag_driver, np.log10(lf_driver), np.log10(np.exp(1))*lf_error/lf_driver, fmt='o', color='#1f77b4', label="z=0 (Driver et al. (2012))")
    ax1.set_xlabel("$M_r$ - $5\log(h)$ (AB)", fontsize=14)
    ax1.set_ylabel("$\log_{10}(\phi/Mpc^{3}h^{-3}/(0.5$ mag))", fontsize=14)
    plt.tick_params(direction='in')
    ax1.tick_params(width=1.3)
    ax1.tick_params('both',length=10, which='major')
    ax1.tick_params('both',length=5, which='minor')
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(12)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(12)
    #plt.ylim(0, 9e-5)
    leg = plt.legend()
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1)
    #plt.show()
    '''
    
    plt.figure()
    plt.plot(bin_mid, SFR_tot*10**-9/volume, label="SHARK")
    plt.plot(bin_mid, SFR_cosmic, label="Cosmic SFR")
    plt.xlabel("Redshift", fontsize=14)
    plt.ylabel("SFR $M_\odot$ $Mpc^{-3}$ $yr^{-1}$", fontsize=14)
    plt.legend()
    plt.show()
    
    SFR_sum = np.zeros(170)
    data = h5py.File('/Users/s4431433/Documents/codes/group/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/199/0/star_formation_histories.hdf5', 'r')
    lbt = np.array(data.get('lbt_mean'))
    lbt = np.concatenate((lbt, np.array([0])))
    lbt_mid = 0.5*(lbt[1:]+lbt[:-1])
    redshift = np.array(data.get('redshifts'))
    RGW_model = (0.5*(17.3+45)*(1+redshift[136:])**2.7/1.2**2.7)*10**-9
    bin_mid_snaps = 0.5*(redshift[9:-1] + redshift[10:])
    radius_snaps = cosmo.comoving_distance(redshift)
    volume_snaps = (1.6857656436069093*64/41252.96) * (4*np.pi*(radius_snaps[9:-1]**3-radius_snaps[10:]**3)/3)
    
    hf1 = h5py.File('/Users/s4431433/Downloads/all_Merger_rate_COMPAS_subvol_0_v1.hdf5', 'r')
    hf2 = h5py.File('/Users/s4431433/Downloads/all_Merger_rate_COMPAS_subvol_0_v3.hdf5', 'r')
    #hf2 = h5py.File('/Users/s4431433/Documents/codes/all_Merger_rate_COMPAS_subvol_0_new_2.hdf5', 'r')
    #hf3 = h5py.File('/Users/s4431433/Documents/codes/all_Merger_rate_COMPAS_subvol_2.hdf5', 'r')
    hf4 = h5py.File('/Users/s4431433/Downloads/all_Merger_rate_COMPAS_subvol_0.hdf5', 'r')
    RGW_sum = np.zeros(170)
    RGW_sum_2 = np.zeros(170)
    RGW_sum_3 = np.zeros(170)
    SFR_gal_sum = np.zeros(170)
    for i in range(30, 200):
        RGW_sum[i-30] = np.sum(np.array(hf1.get(str('%d/Merger rate ((bulge disk instabilities)' %i)))) + np.sum(np.array(hf1.get(str('%d/Merger rate (bulge mergers)' %i)))) + np.sum(np.array(hf1.get(str('%d/Merger rate (disk)' %i))))
        hf5 = h5py.File(str('/Users/s4431433/Documents/codes/group/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/%d/0/star_formation_histories.hdf5' %i), 'r')
        #index = np.where(np.array(hf5.get('galaxies/mstars_disk')) + np.array(hf5.get('galaxies/mstars_bulge')) > 0)[0]
        RGW_sum_2[i-30] = np.sum(np.array(hf2.get(str('%d/Merger rate (bulge disk instabilities)' %i)))) + np.sum(np.array(hf2.get(str('%d/Merger rate (disk)' %i))))
        #SFR_gal_sum[i-30] = np.sum(np.array(hf5.get('bulges_diskins/star_formation_rate_histories'))[:,-1]) + np.sum(np.array(hf5.get('bulges_mergers/star_formation_rate_histories'))[:,-1]) + np.sum(np.array(hf5.get('disks/star_formation_rate_histories'))[:,-1])
        #print(np.where(np.array(hf2.get(str('%d/Merger rate (disk)' %i)))<0)[0])
        #RGW_sum[2][i-115] = np.sum(np.array(hf3.get(str('%d/Merger rate (bulge disk instabilities)' %i)))) + np.sum(np.array(hf3.get(str('%d/Merger rate (bulge mergers)' %i)))) + np.sum(np.array(hf3.get(str('%d/Merger rate (disk)' %i))))
        RGW_sum_3[i-30] = np.sum(np.array(hf4.get(str('%d/Merger rate (bulge disk instabilities)' %i)))) + np.sum(np.array(hf4.get(str('%d/Merger rate (disk)' %i))))
        if i == 199:
            print(np.array(hf2.get(str('%d/Merger rate (disk)' %i))) - np.array(hf4.get(str('%d/Merger rate (disk)' %i))))
    #print(RGW_sum-RGW_sum_2)
    #print(RGW_sum*(0.7**3)/((210**3/32)*Normalisation_factor))
    #print(SFR_gal_sum[-1])
    print(RGW_sum_2)
    print(RGW_sum_3)
    plt.figure()
    plt.plot(lbt_mid[10:], RGW_sum*(0.7**3)/((210**3/32)*Normalisation_factor), label="snapshots")
    plt.plot(lbt_mid[10:], RGW_sum_2*(0.7**3)/((210**3/32)*Normalisation_factor), label="mstars_metal/mstars")
    #plt.plot(lbt_mid[10:], SFR_gal_sum*(0.7**3)/((210**3/32)*Normalisation_factor), label="galaxies file")
    plt.plot(lbt_mid[10:], RGW_sum_3*(0.7**3)/((210**3/32)*Normalisation_factor), label="mgas_metals/mgas")
    plt.plot(lbt_mid[136:], RGW_model, linestyle='dashed', label="GWTC-3")
    #plt.plot(lbt[10:], RGW_sum)
    #plt.errorbar(lbt[95:], RGW_sum.mean(axis=0)*(0.7**3)/((210**3/32)*Normalisation_factor), yerr=RGW_sum.std(axis=0)*(0.7**3)/((210**3/32)*Normalisation_factor), capsize=5)
    plt.xlabel("Lookback Time (Gyr)", fontsize=14)
    plt.ylabel("$R_{GW}$ $(Mpc/h)^{-3}$ $yr^{-1}$", color = 'k', fontsize=14)
    plt.legend()
    plt.show()
    
    RGW_model = 19.3*(1+bin_mid)**1.3
    RGW_model_2 = 40.39*(1+bin_mid)**1.56
    RGW_model_upper = 34.4*(1+bin_mid)**3.4
    RGW_model_lower = 10.3*(1+bin_mid)**-0.8
    print(RGW_sum_2)
    fig, ax1 = plt.subplots()
    #ax1.plot(dist.value*10**-3, RGW_num*10**9/volume.value, color = 'b')
    #ax1.set_ylabel("$N_{gal}/Gpc^{-3}$", color = 'b')
    RGW_model_upper = 34.4*(1+bin_mid)**3.4
    RGW_model_lower = 10.3*(1+bin_mid)**-0.8
    ax1.fill_between(bin_mid, RGW_model_lower, RGW_model_upper, color='grey', alpha =0.25)
    ln = ax1.plot(bin_mid,RGW_model, color="grey", linewidth=3, label='Uncertainty')
    #lns0 = ax1.plot(bin_mid, RGW_model, color='k', linestyle='--', label= r"Abbott $\it{et}$ $\it{al.}$ (2021)")
    #lns0 = ax1.plot(bin_mid, RGW_model, color='k', linestyle='--', label= "Observation")
    #lns1 = ax1.plot(bin_mid, RGW_tot*10**9/volume, color = 'r')
    lns1 = ax1.plot(redshift[95:], RGW_sum.mean(axis=0)/Normalisation_factor, color = 'r')
    Cdata = np.load('COMPAS_fractions.npz')
    Cdata1 = np.load('COMPAS_fractions_5.npz')
    frac = np.mean(Cdata1["y0"])/np.mean(Cdata["y0"])
    #lns2 = ax1.plot(bin_mid, frac*RGW_tot*10**9/volume, color = 'b')
    Cdata2 = np.load('COMPAS_fractions_3.npz')
    frac = np.mean(Cdata2["y0"])/np.mean(Cdata["y0"])
    #lns3 = ax1.plot(bin_mid, frac*RGW_tot*10**9/volume, color = 'g')
    #lns2 = ax1.plot(bin_mid, RGW_tot_1*10**9/volume, color = 'r', label='$log(sSFR)<-10$')
    #lns3 = ax1.plot(bin_mid, RGW_tot_2*10**9/volume,  color = 'r', linestyle='dashed', label='$log(sSFR) \geq -10$')
    print(RGW_tot*10**9/(volume*(210/0.68)**3/64))
    #print(RGW_sum*10**9/(Normalisation_factor*(210/0.68)**3/64))
    #ax1.set_xlabel("Distance (Gpc)")
    #ax1.set_ylim(0,400)
    ax1.set_xlabel("Redshift", fontsize=14)
    ax1.set_ylabel("$R_{GW}$ $Gpc^{-3}$ $yr^{-1}$", color = 'k', fontsize=14)
    ax2 = ax1.twinx()
    #lns2 = ax2.plot(bin_mid, Normalisation_factor*SFR_tot/volume.value,  color = 'b', label="SHARK SFR") # converted to M_\odot yr^-1 Mpc^-3
    #lns4 = ax2.plot(bin_mid, SFR_cosmic*10**9,  color = 'g', label="Cosmic SFR")
    ax2.set_ylabel("SFR $M_\odot$ $Gpc^{-3}$ $yr^{-1}$", color = 'k', fontsize=14)
    ax2.set_ylim(0, 8e7)
    #ax2.set_ylim(None, 4*10**7)
    lns = ln
    ln = ln.pop(0)
    ln.remove()
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc="upper left")
    #plt.show()
    
    
    for i in range(30, 200):
        #print(i)
        data = h5py.File(str('/Users/s4431433/Documents/codes/group/pawsey0119/clagos/SHARK_Out/medi-SURFS/Shark-TreeFixed-ReincPSO-kappa0p002/%d/0/star_formation_histories.hdf5' %i), 'r')
        sfr1 = np.array(data.get('bulges_diskins/star_formation_rate_histories'))
        sfr2 = np.array(data.get('bulges_mergers/star_formation_rate_histories'))
        sfr3 = np.array(data.get('disks/star_formation_rate_histories'))
        sfr = sfr1+sfr2+sfr3
        SFR_sum[i-30] = np.sum(sfr[:,-1])
    '''
    cosmo = FlatLambdaCDM(H0=70, Om0=0.3) #changed om0 from 0.27
    data = np.load('all_subvols.npz')
    SFR = data['y1']
    zobs = data['y0']
    SFR_tot, bin_edges = np.histogram(zobs, bins = N, weights = SFR)
    
    data = np.load('all_SFR_snaps_2.npz')
    SFR_sum = data['y0']/10**9
    SFR_sum_mean = SFR_sum.mean(axis=0)
    SFR_sum_err = SFR_sum.std(axis=0)
    print(np.shape(SFR_sum_mean))
    
    data2 = np.load('all_SFR_snaps.npz')
    SFR_sum2 = data2['y0']
    SFR_sum_mean2 = SFR_sum2.mean(axis=0)
    SFR_sum_err2 = SFR_sum2.std(axis=0)
    print(np.shape(SFR_sum_mean2))
    '''
    plt.figure()
    plt.plot(lbt[10:], (SFR_sum_mean-SFR_sum_mean2)*(0.7**2)/(210**3/64))
    plt.xlabel("Lookback Time (Gyr)")
    plt.show()
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    radius = cosmo.comoving_distance(bin_edges)
    volume = (1.6857656436069093*64/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    '''
    #https://www.slac.stanford.edu/~digel/ObservingStrategy/whitepaper/LSST_Observing_Strategy_White_Paper.pdf
    CV = np.array([0.07, 0.05, 0.04, 0.05, 0.06, 0.06, 0.09, 0.07, 0.06, 0.07, 0.05, 0.06, 0.06, 0.07, 0.04, 0.04, 0.03, 0.05, 0.04])
    z_min = np.array([0.01, 0.2, 0.4, 0.6, 0.8, 0.05, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.7, 2.5, 3.5, 0.92, 1.62, 2.08, 1.9, 2.7, 3.8, 4.9, 5.9, 7.0, 7.9, 7.0, 8.0, 0.03, 0.03, 0.4, 0.7, 1.0, 1.3, 1.8, 0.4, 0.7, 1.0, 1.3, 1.8, 0, 0.3, 0.45, 0.6, 0.8, 1, 1.2, 1.7, 2, 2.5, 3])
    z_max = np.array([0.1, 0.4, 0.6, 0.8, 1.2, 0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.7, 2.5, 3.5, 4.5, 1.33, 1.88, 2.37, 2.7, 3.4, 8, 4.9, 5.9, 7.0, 7.9, 7.0, 8.0, 0.03, 0.03, 0.7, 1.0, 1.3, 1.8, 2.3, 0.7, 1.0, 1.3, 1.8, 2.3, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.7, 2.0, 2.5, 3.0, 4.2])
    z_new =0.5*(z_min+z_max)
    print(len(z_new))
    SFR_obs = 0.63*10**np.array([-1.82, -1.5, -1.39, -1.20, -1.25, -1.77, -1.75, -1.55, -1.44, -1.24, -0.99, -0.94, -0.95, -0.75, -1.04, -1.69, -1.02, -0.75, -0.87, -0.75, -0.97, -1.29, -1.42, -1.65, -1.79, -2.09, -2, -2.21, -1.72, -1.95, -1.34, -0.96, -0.89, -0.91, -0.89, -1.22, -1.1, -0.96, -0.94, -0.8, -1.64, -1.42, -1.32, -1.14, -0.94, -0.81, -0.84, -0.86, -0.91, -0.86, -1.36]) # rescale SFR and then take log
    print(len(SFR_obs))
    error_new = 10**np.array([(0.09,0.02), (0.05, 0.05), (0.15, 0.08), (0.31, 0.13), (0.31, 0.13), (0.08,0.09), (0.18, 0.18), (0.12, 0.12), (0.1, 0.1), (0.1, 0.1), (0.09, 0.08), (0.09, 0.09), (0.15, 0.08), (0.49, 0.09), (0.26, 0.15), (0.22, 0.32), (0.08, 0.08), (0.12, 0.12), (0.09, 0.09), (0.09, 0.11), (0.11, 0.15), (0.05, 0.05), (0.06, 0.06), (0.08, 0.08), (0.1, 0.1), (0.11, 0.11), (0.1, 0.11), (0.14, 0.14), (0.02, 0.03), (0.2, 0.2), (0.22, 0.11), (0.15, 0.19), (0.27, 0.21), (0.17, 0.21), (0.21, 0.25), (0.08, 0.11), (0.1, 0.13), (0.13, 0.2), (0.13, 0.18), (0.18, 0.15), (0.09, 0.11), (0.03, 0.04), (0.05, 0.05), (0.06, 0.06), (0.05, 0.06), (0.04, 0.05), (0.04, 0.04), (0.02, 0.03), (0.09, 0.12), (0.15, 0.23), (0.23, 0.5)]).T
    print(np.shape(error_new))
    fig, ax1 = plt.subplots()
    #ln, = ax1.plot(cosmo.age(0)-cosmo.age(bin_mid), SFR_tot*(0.7/0.678)**3/(10**9*volume.value),  color = 'b')
    #ax1.errorbar(lbt[10:], SFR_sum_mean*(0.7/0.678**2)**-2/(210**3/64), yerr=SFR_sum_err*(0.7/0.678**2)**-2/(210**3/64), capsize=2, color = 'b', label="SHARK SFR")
    ln, = ax1.plot(lbt[10:], np.log10(SFR_sum_mean*(0.7**2)/(210**3/64)), color = 'b')
    lerr = ax1.fill_between(lbt[10:], np.log10(SFR_sum_mean*(0.7**2)/(210**3/64)-SFR_sum_err*(0.7**2)/(210**3/64)), np.log10(SFR_sum_mean*(0.7**2)/(210**3/64)+SFR_sum_err*(0.7**2)/(210**3/64)), color="b", alpha=0.5)
    #ax1.plot(lbt[10:], np.log10(SFR_sum_mean2*(0.7**2)/(210**3/64)), color = 'g', label="sfr histories files")
    #ax1.fill_between(lbt[10:], np.log10(SFR_sum_mean2*(0.7**2)/(210**3/64)-SFR_sum_err2*(0.7**2)/(210**3/64)), np.log10(SFR_sum_mean2*(0.7**2)/(210**3/64)+SFR_sum_err2*(0.7**2)/(210**3/64)), color="g", alpha=0.5)
    #ln2, = ax1.plot(cosmo.age(0)-cosmo.age(bin_mid), SFR_cosmic,  color = 'g')
    ln2, = ax1.plot(lbt, np.log10(SFR_cosmic),  color = 'g')
    Age = cosmo.age(0)-cosmo.age(z_mid)
    Age_new = cosmo.age(0)-cosmo.age(z_new)
    print(Age_new.value)
    print(SFR_obs)
    lerr1 = ax1.fill_between(Age.value, sfr_vals-CV, sfr_vals+CV, color="r", alpha=0.5)
    ln1 = ax1.scatter(Age.value, sfr_vals, color='r')
    ax1.set_ylabel("$\log_{10}$(CSFRD/$M_\odot$ $yr^{-1}$ $h_{70}^{-3}$ $Mpc^{-3}$)", color = 'k', fontsize=14)
    ln3 = ax1.errorbar(Age_new.value, np.log10(SFR_obs), yerr=np.log10(error_new), fmt='.', capsize=3, alpha=0.5, color="grey")
    #plt.xlabel("Redshift", fontsize=14)
    ax1.set_xlabel("Lookback time (Gyr)", fontsize=14)
    ax1.tick_params(width=1.3)
    ax1.tick_params(direction='in')
    ax1.tick_params('both',length=10, which='major')
    ax1.tick_params('both',length=5, which='minor')
    for axis in ['top','left','bottom','right']:
        ax1.spines[axis].set_linewidth(1.3)
    for tick in ax1.xaxis.get_ticklabels():
        tick.set_fontsize(12)
    for tick in ax1.yaxis.get_ticklabels():
        tick.set_fontsize(12)
    #plt.yscale("log")
    leg = ax1.legend([(ln, lerr), (ln1, lerr1), ln2, ln3], ["SHARK SFR", "Driver et al. (2018)", "Madau & Dickinson (2014) fit", "Madau & Dickinson (2014) data \ncompilation"], loc='upper left')
    leg.get_frame().set_edgecolor('k')
    leg.get_frame().set_linewidth(1)
    plt.show()
    '''
    RGW_sum = np.zeros((len(subvol_list), N))
    RGW_sum_2 = np.zeros((len(subvol_list), N))
    RGW_err_new = np.zeros(N)
    RGW_ave_new = np.zeros(N)
    RGW_ratio = np.zeros((len(subvol_list), N))
    for i in range(len(subvol_list)):
        RGW = np.load(str('properties_subvol_%d.npz' % subvol_list[i]))['y4']
        RGW_tot_old, _ = np.histogram(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6'], bins = bin_edges, weights = RGW)
        index = np.where(np.logical_and(np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']-np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y3']<0.45, np.load(str('photometry_subvol_%d.npz' % subvol_list[i]))['y1']<17))[0]
        RGW_tot_new, _ = np.histogram(np.load(str('properties_subvol_%d.npz' %subvol_list[i]))['y6'][index], bins = bin_edges, weights = RGW[index])
        RGW_sum[i,:] = RGW_tot_old
        RGW_sum_2[i,:] = RGW_tot_new
        RGW_ratio[i,:] = RGW_sum_2[i,:]/RGW_sum[i,:]
    RGW_ratio_tot = np.zeros((1000, N))
    for i in range(1000):
        for j in range(N):
            #boot = resample(RGW_sum[:,j], replace=True, n_samples=10, random_state=1)
            boot = np.random.choice(RGW_sum[:,j], size=10, replace=True)
            boot_2 = np.random.choice(RGW_sum_2[:,j], size=10, replace=True)
            #boot_2 = resample(RGW_sum_2[:,j], replace=True, n_samples=10, random_state=1)
            RGW_vol_tot[i,j] = np.sum(boot)
            RGW_vol_tot_2[i,j] = np.sum(boot_2)
            RGW_ratio_tot[i,j] = RGW_vol_tot_2[i,j]/RGW_vol_tot[i,j]
    
    RGW_error_1 = np.zeros(N)
    RGW_err_new = np.std(RGW_ratio_tot, axis=0)
    RGW_ave_new = np.mean(RGW_ratio_tot, axis=0)
    '''
    #print(RGW_ave_new)
    #print(RGW_ratio)
    #print(RGW_err_new)
    '''
    #for i in range(len(RGW_tot)):
        #index_2 = np.where(np.logical_and(z_all[index] >= bin_edges[i], z_all[index] < bin_edges[i+1]))[0]
        #RGW_tot_new = np.array(RGW_all[index][index_2])
        #RGW_sum.append(np.sum(RGW_tot_new))
    
    colours = ['#d73027', '#fc8d59', '#fee090', '#91bfdb', '#4575b4', '#bae4bc']
    fig, ax = plt.subplots()

    index = np.where(np.logical_and(J_mag_all-K_mag_all<0.45, J_mag_all<17))[0]
    RGW_sum, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    RGW_err_new, RGW_ave_new = RGW_error_func(subvol_list, bin_edges, N, 1)
    #RGW_ratio_err = (RGW_sum/RGW_tot)*np.sqrt((RGW_err_new/RGW_sum)**2 + (RGW_err/RGW_tot)**2)
    plt.plot(dist*10**-3, RGW_sum/RGW_tot, alpha=0.5, label= "4MOST HS")
    ax.errorbar(dist.value*10**-3, RGW_sum/RGW_tot, yerr=RGW_err_new, capsize=2, fmt="none", ecolor="grey")
    
    index = np.where(np.logical_and(J_mag_all>16, J_mag_all<18))[0]
    RGW_err_new, RGW_ave_new = RGW_error_func(subvol_list, bin_edges, N, 2)
    RGW_sum, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    ax.plot(dist*10**-3, RGW_sum/RGW_tot, label= "4MOST CRS")
    trans1 = Affine2D().translate(+0.02, 0.0) + ax.transData
    ax.errorbar(dist.value*10**-3, RGW_sum/RGW_tot, yerr=RGW_err_new, capsize=2, fmt="none", ecolor="k", transform = trans1)
    
    index = np.where(np.logical_and(np.logical_and(np.logical_and(g_mag_all - r_mag_all > -1, g_mag_all - r_mag_all < 4), np.logical_and(r_mag_all - z_mag_all < 4, r_mag_all - z_mag_all > -1)), r_mag_all < 19.5))[0]
    RGW_err_new, RGW_ave_new = RGW_error_func(subvol_list, bin_edges, N, 3)
    RGW_sum, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    plt.plot(dist*10**-3, RGW_sum/RGW_tot, label= "DESI BGS")
    trans1 = Affine2D().translate(+0.01, 0.0) + ax.transData
    plt.errorbar(dist.value*10**-3, RGW_sum/RGW_tot, yerr=RGW_err_new, capsize=2, fmt="none", ecolor="grey", transform = trans1)
    
    index = np.where(np.logical_and(g_mag_all < 23.8, np.logical_and(r_mag_all < 23.1, z_mag_all < 22.5)))[0]
    RGW_err_new, RGW_ave_new = RGW_error_func(subvol_list, bin_edges, N, 4)
    RGW_sum, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    plt.plot(dist*10**-3, RGW_sum/RGW_tot, label= "DESI LIS")
    trans1 = Affine2D().translate(-0.01, 0.0) + ax.transData
    plt.errorbar(dist.value*10**-3, RGW_sum/RGW_tot, yerr=RGW_err_new, capsize=2, fmt="none", ecolor="k", transform = trans1)
    
    index = np.where(np.logical_and(u_mag_all<25.4, np.logical_and(y_mag_all < 24.4, np.logical_and(g_mag_all<27.0, np.logical_and(r_mag_all<27.1, np.logical_and(i_mag_all<26.4, z_mag_all<25.2))))))[0]
    RGW_err_new, RGW_ave_new = RGW_error_func(subvol_list, bin_edges, N, 5)
    RGW_sum, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    plt.plot(dist*10**-3, RGW_sum/RGW_tot, label= "LSST")
    trans1 = Affine2D().translate(-0.02, 0.0) + ax.transData
    plt.errorbar(dist.value*10**-3, RGW_sum/RGW_tot, yerr=RGW_err_new, capsize=2, fmt="none", ecolor="grey", transform = trans1)
    
    WALLABY_data = np.load("WALLABY_data.npz")
    RGW_ratio = WALLABY_data["y0"]
    RGW_err = WALLABY_data["y1"]
    plt.plot(dist*10**-3, RGW_ratio, label= "WALLABY")
    trans1 = Affine2D().translate(-0.03, 0.0) + ax.transData
    plt.errorbar(dist.value*10**-3, RGW_ratio, yerr=RGW_err, capsize=2, fmt="none", ecolor="k", transform = trans1)
    #plt.xlim(0,400)
    #plt.xscale("log")
    plt.legend()
    plt.xlabel("Distance (Gpc)", fontsize=13)
    plt.ylabel("$\mathcal{R}_{GW, survey}/\mathcal{R}_{GW, SHARK}$", fontsize=13)
    plt.show()
    '''
    W1 = np.array(ph_data["y14"])
    W2 = np.array(ph_data["y15"])
    ######################################################################################
    plt.figure()
    ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
    RGW_tot, bin_edges = np.histogram(z_all, bins = 30, weights = RGW_all)
    
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    RGW_tot_shark = RGW_tot
    dist = cosmo.comoving_distance(bin_mid)
    dist = dist*10**-3 #Gpc
    #ax1.fill_between(bin_mid, RGW_model_lower, RGW_model_upper, color='grey', alpha =0.25)
    volume = Normalisation_factor*(1.6857656436069093/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    ax1.plot(bin_mid, RGW_tot_shark*10**9/volume.value, color = 'b', label= "SHARK")
    #ax1.plot(bin_mid, RGW_model, color='k', linestyle='--', label= r"Abbott $\it{et}$ $\it{al.}$ (2021)")
    #ax2 = ax1.twinx()
    #ax2.plot(dist, SFR_tot/volume.value, color = "DeepPink")
    #ax2.set_ylabel("SFR $M_\odot$ $Gpc^{-3}$ $yr^{-1}$", color = "DeepPink")
    
    #index = np.where(np.logical_and(np.logical_and(np.logical_and(J_mag_all>16, J_mag_all<18.25), np.logical_and(J_mag_all-W1>1.6*(J_mag_all-K_mag_all)-1.6, J_mag_all-W1<1.6*(J_mag_all-K_mag_all)-0.5)), np.logical_and(J_mag_all-W1>-2.5*(J_mag_all-K_mag_all)+0.1, J_mag_all-W1<-0.5*(J_mag_all-K_mag_all)+0.1)))[0] # BG
    index = np.where(np.logical_and(np.logical_and(np.logical_and(J_mag_all>18, J_mag_all<19.5), np.logical_and(J_mag_all-K_mag_all<1.5, J_mag_all-W1<1.2)), np.logical_and(J_mag_all-W1>0.5*(J_mag_all-K_mag_all)+0.05, np.logical_and(J_mag_all-W1>-0.5*(J_mag_all-K_mag_all)+0.1, J_mag_all-W1<0.5*(J_mag_all-K_mag_all)+0.))))[0] #LRG
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color='darkorange', label= "4MOST CRS")
    print(np.sum(RGW_all[index]))
    print("4MOST CRS", len(index)/1.6857656436069093)
    
    index = np.where(np.logical_and(J_mag_all-K_mag_all<0.45, J_mag_all<18))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'g', label= "4MOST HS")
    print(np.sum(RGW_all[index]))
    print("4MOST HS", len(index)/1.6857656436069093)
    
    index = np.where(np.logical_and(g_mag_all < 27.0, np.logical_and(r_mag_all < 27.1, z_mag_all < 25.2)))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'r', label= "LSST WFD")
    print(np.sum(RGW_all[index]))
    print("LSST", len(index)/1.6857656436069093)
    
    index = np.where(np.logical_and(np.logical_and(np.logical_and(g_mag_all - r_mag_all > -1, g_mag_all - r_mag_all < 4), np.logical_and(r_mag_all - z_mag_all < 4, r_mag_all - z_mag_all > -1)), r_mag_all < 19.5))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'darkviolet', label= "DESI BGS")
    print(np.sum(RGW_all[index]))
    print("DESI BGS", len(index)/1.6857656436069093)
    
    index = np.where(np.logical_and(g_mag_all < 24.3, np.logical_and(r_mag_all < 23.7, z_mag_all < 23.3)))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'hotpink', label= "DESI LIS")
    print(np.sum(RGW_all[index]))
    print("DESI LIS", len(index)/1.6857656436069093)
    
    #index = np.where(np.logical_and(u_mag_all<23.14, np.logical_and(y_mag_all < 21.57, np.logical_and(np.logical_and(g_mag_all<24.47, r_mag_all<24.16), np.logical_and(i_mag_all<23.40, z_mag_all<22.23)))))[0]
    #index = np.where(np.logical_and(u_mag_all<25.4, np.logical_and(y_mag_all < 24.4, np.logical_and(g_mag_all<27.0, np.logical_and(r_mag_all<27.1, np.logical_and(i_mag_all<26.4, z_mag_all<25.2))))))[0] ##### LSST
  
    index = np.where(np.logical_and(np.logical_and(g_mag_all<23.6, g_mag_all>20), np.logical_and(np.logical_and(r_mag_all-z_mag_all<1.6, r_mag_all-z_mag_all>0.3), np.logical_and(g_mag_all-r_mag_all<1.15*(r_mag_all-z_mag_all)-0.35, g_mag_all-r_mag_all<-1.2*(r_mag_all-z_mag_all)+1.6))))[0]
    print(len(index))
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'gold', label= "DESI ELG")
    print(np.sum(RGW_all[index]))
    print("DESI ELG", len(index)/1.6857656436069093)
 
    #index = np.where(np.logical_and(np.logical_and(z_mag_all-W1>0.8*(r_mag_all-z_mag_all)-0.65, r_mag_all-W1>1.85), np.logical_and(r_mag_all-z_mag_all>0.45*(z_mag_all-16.69), r_mag_all-z_mag_all>0.19*(z_mag_all-13.68))))[0]
    #index = np.where(np.logical_and(np.logical_and(z_mag_all-W1>0.8*(r_mag_all-z_mag_all)-0.65, np.logical_and(g_mag_all-W1>2.67, g_mag_all-r_mag_all>1.45)), np.logical_and(r_mag_all-z_mag_all>0.45*(z_mag_all-16.69), r_mag_all-z_mag_all>0.19*(z_mag_all-13.68))))[0]
   
    #index = np.where(np.logical_and(np.logical_and(z_mag_all-W1>0.8*(r_mag_all-z_mag_all)-0.65, np.logical_or(np.logical_and(g_mag_all-W1>2.67, g_mag_all-r_mag_all>1.45), r_mag_all-W1>1.85)), np.logical_and(r_mag_all-z_mag_all>0.45*(z_mag_all-16.69), r_mag_all-z_mag_all>0.19*(z_mag_all-13.68))))[0]
    
    index = np.where(np.logical_and(np.logical_and(z_mag_all-W1>0.8*(r_mag_all-z_mag_all)-0.6, np.logical_or(np.logical_and(g_mag_all-W1>2.6, g_mag_all-r_mag_all>1.4), r_mag_all-W1>1.8)), np.logical_and(np.logical_and(r_mag_all-z_mag_all>(z_mag_all-16.83)*0.45, r_mag_all-z_mag_all>(z_mag_all-13.80)*0.19), r_mag_all-z_mag_all>0.7)))[0] #Southern
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'maroon', label= "DESI LRG")
    print(np.sum(RGW_all[index]))
    print("DESI LRG", len(index)/1.6857656436069093)
    
    f_grz = (flux_func(g_mag_all, "g") + 0.8*flux_func(r_mag_all, "r") + 0.5*flux_func(z_mag_all, "z"))/2.3
    flux_W = 0.75*flux_func(W1, "W1") + 0.25*flux_func(W2, "W2")
    print(f_grz)
    print(flux_W)
    m_sun, F_sun = -25.18, 1584.71
    grz = -2.5*np.log10(f_grz/F_sun) + m_sun
    W = -2.5*np.log10(flux_W/F_sun) + m_sun
    print(grz)
    print(W)
    #index = np.where(np.logical_and(np.logical_and(np.logical_and(grz>17, grz-W >= g_mag_all-z_mag_all-1) , np.logical_and(g_mag_all-r_mag_all<1.3, r_mag_all>17.5)), np.logical_and(r_mag_all-z_mag_all<1.1, r_mag_all-z_mag_all>-0.4)))[0]
    #index = np.where(np.logical_and(np.logical_and(np.logical_and(f_grz<10**((22.5-17)/2.5), flux_func(g_mag_all, "g")*flux_W > 10**(-1.3/2.5)*f_grz*flux_func(z_mag_all, "z")) , np.logical_and(flux_func(r_mag_all, "r")<10**(1.3/2.5)*flux_func(g_mag_all, "g"), flux_func(r_mag_all, "r")<10**((22.5-17.5)/2.5))), np.logical_and(flux_func(z_mag_all, "z")<10**(1.1/2.5)*flux_func(r_mag_all, "r"), flux_func(z_mag_all, "z")>10**(-0.4/2.5)*flux_func(r_mag_all, "r"))))[0]
    #index = np.where(np.logical_and(np.logical_and(f_grz<10**((22.5-17)/2.5), flux_func(g_mag_all, "g")*flux_W > 10**(-1.3/2.5)*f_grz*flux_func(z_mag_all, "z")), np.logical_and(flux_func(r_mag_all, "r")<10**(1.3/2.5)*flux_func(g_mag_all, "g"), flux_func(g_mag_all, "g")))
    #RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #volume = (1.6857656436069093*10/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    #dist = cosmo.comoving_distance(bin_mid)
    #ax1.plot(bin_mid, RGW_tot*10**9/volume.value, color = 'dodgerblue', label= "DESI QSO")
    #print(np.sum(RGW_all[index]))
    #print("DESI QSO", len(index)/1.6857656436069093)
    
    #plt.xscale("log")
    ax1.legend(bbox_to_anchor=(2.2,0.5), loc="center left")
    #ax1.set_ylim(0, 400)
    ax1.set_xlabel("Redshift", fontsize=14)
    ax1.set_ylabel("$\mathcal{R}_{GW}$ ($Gpc^{-3}$ $yr^{-1}$)", fontsize=14)
    
    ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
    index = np.where(np.logical_and(J_mag_all>16, J_mag_all<18))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    volume = (1.6857656436069093/41252.96) * (4*np.pi*(radius[1:]**3-radius[:-1]**3)/3)
    ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color='darkorange', label= "4MOST CRS")
    index = np.where(np.logical_and(J_mag_all-K_mag_all<0.45, J_mag_all<17))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'g', label= "4MOST HS")
    index = np.where(np.logical_and(np.logical_and(np.logical_and(g_mag_all - r_mag_all > -1, g_mag_all - r_mag_all < 4), np.logical_and(r_mag_all - z_mag_all < 4, r_mag_all - z_mag_all > -1)), r_mag_all < 19.5))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'darkviolet', label= "DESI BGS")
    index = np.where(np.logical_and(np.logical_and(z_mag_all-W1>0.8*(r_mag_all-z_mag_all)-0.65, r_mag_all-W1>1.85), np.logical_and(r_mag_all-z_mag_all>0.45*(z_mag_all-16.69), r_mag_all-z_mag_all>0.19*(z_mag_all-13.68))))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'maroon', label= "DESI LRG")
    #index = np.where(np.logical_and(np.logical_and(np.logical_and(grz>17, grz-W >= g_mag_all-z_mag_all-1) , np.logical_and(g_mag_all-r_mag_all<1.3, r_mag_all>17.5)), np.logical_and(r_mag_all-z_mag_all<1.1, r_mag_all-z_mag_all>-0.4)))[0]
    #RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    #radius = cosmo.comoving_distance(bin_edges)
    #bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    #ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'dodgerblue', label= "DESI QSO")
    index = np.where(np.logical_and(np.logical_and(g_mag_all<23.6, g_mag_all>20), np.logical_and(np.logical_and(r_mag_all-z_mag_all<1.6, r_mag_all-z_mag_all>0.3), np.logical_and(g_mag_all-r_mag_all<1.15*(r_mag_all-z_mag_all)-0.35, g_mag_all-r_mag_all<-1.2*(r_mag_all-z_mag_all)+1.6))))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax2.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'gold', label= "DESI ELG")
    ax2.set_ylim(0,1)
    text_kwargs = dict(ha='center', va='center', fontsize=13, color='k')
    ax2.text(2.5, 0.8, 'Spectroscopic', **text_kwargs)
    ax2.set_ylabel("$\mathcal{R}_{GW, survey}/\mathcal{R}_{GW, SHARK}$", fontsize=10)
    
    
    ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
    index = np.where(np.logical_and(g_mag_all < 24.3, np.logical_and(r_mag_all < 23.7, z_mag_all < 23.3)))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax3.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'hotpink', label= "DESI LIS")
    index = np.where(np.logical_and(g_mag_all < 27.0, np.logical_and(r_mag_all < 27.1, z_mag_all < 25.2)))[0]
    RGW_tot, _ = np.histogram(z_all[index], bins = bin_edges, weights = RGW_all[index])
    radius = cosmo.comoving_distance(bin_edges)
    bin_mid = 0.5*(bin_edges[:-1] + bin_edges[1:])
    ax3.plot(bin_mid, RGW_tot/RGW_tot_shark, color = 'r', label= "LSST WFD")
    text_kwargs = dict(ha='center', va='center', fontsize=13, color='k')
    ax3.text(2.5, 0.9, 'Photometric', **text_kwargs)
    ax3.set_xlabel("Redshift", fontsize = 14)
    ax3.set_ylabel("$\mathcal{R}_{GW, survey}/\mathcal{R}_{GW, SHARK}$", fontsize=10)
    plt.show()
    
    #index = np.where(grz-W >= g_mag_all-z_mag_all-1)[0]
    index = np.where(np.logical_and(np.logical_and(np.logical_and(grz>17, grz-W >= g_mag_all-z_mag_all-1) , np.logical_and(g_mag_all-r_mag_all<1.3, r_mag_all>17.5)), np.logical_and(r_mag_all-z_mag_all<1.1, r_mag_all-z_mag_all>-0.4)))[0]
    print(len(index))
    plt.figure()
    plt.scatter(g_mag_all[index]-z_mag_all[index], grz[index]-W[index])
    plt.show()
    '''
    index = np.where(np.logical_and(ph_data['y1']>16, ph_data['y1']<18))[0]
    plt.figure()
    plt.hist(data['y6'], bins=30, weights = data['y4'], alpha = 0.5)
    plt.hist(data['y6'][index], bins=30, weights = data['y4'][index], alpha = 0.5, label= "4MOST CRS")
    plt.xlabel("Observed redshift")
    plt.ylabel("Total $R_\mathrm{GW, obs} [\mathrm{yr^{-1}}]$")
    plt.legend()
    
    index = np.where(np.logical_and(ph_data_2['y1']>16, ph_data_2['y1']<18))[0]
    plt.figure()
    plt.hist(data_2['y6'], bins=30, weights = data_2['y4'], alpha = 0.5)
    plt.hist(data_2['y6'][index], bins=30, weights = data_2['y4'][index], alpha = 0.5, label= "4MOST CRS")
    plt.xlabel("Observed redshift")
    plt.ylabel("Total $R_\mathrm{GW, obs} [\mathrm{yr^{-1}}]$")
    plt.legend()
    
    index = np.where(np.logical_and(ph_data_3['y1']>16, ph_data_3['y1']<18))[0]
    plt.figure()
    plt.hist(data_3['y6'], bins=30, weights = data_3['y4'], alpha = 0.5)
    plt.hist(data_3['y6'][index], bins=30, weights = data_3['y4'][index], alpha = 0.5, label= "4MOST CRS")
    plt.xlabel("Observed redshift")
    plt.ylabel("Total $R_\mathrm{GW, obs} [\mathrm{yr^{-1}}]$")
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    counts,xbins,ybins = np.histogram2d(log_M, log_Z,bins=30)
    print(np.amin(log_Z), np.amax(log_Z))
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
    #reduce_C_function=np.sum,
    my_mean_func = lambda x: np.log10(np.mean(x))
    my_sum_func  = lambda x: np.log10(np.sum(x))
    cax = plt.hexbin(log_M, log_Z, C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-5.0, vmax=0)
    #print(cax)
    #plt.vlines(10.2, -5, 0, 'r', '--')
    #plt.hlines(-1.5, 7, 12, 'r', '--')
    #plt.xlim(7, 12)
    #plt.ylim(-4.5, -0.2)
    #plt.fill_between([10.2,12], -1.5, 1, color='grey', alpha =0.25)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.ylabel("$\log(Z)$", fontsize=12.5)
    #plt.ylabel("$\log(SFR [M_{\odot}/yr])$")
    plt.xlabel("$\log(M_* [M_{\odot}])$", fontsize=12.5)
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%.2f}$"))
    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", fontsize=12.5, rotation=270, labelpad = 18)
    #plt.savefig('/Users/s4431433/Documents/codes/Paper plots subvolume 1/hexbin_ZvM_mean.png')
    plt.show()
    
    index = np.where(np.logical_and(log_M > 9.6, log_sSFR > -15))[0]
    print(len(index))
    R_sum_cutoff = np.sum(data['y4'][index])
    print(R_sum_cutoff/len(index))
    print(np.sum(data['y4'])/47316)

    
    with PdfPages('COLOR_COLOR.pdf') as pdf:
        #pdf = PdfPages('multipage_pdf.pdf')
        for i in range(8):
            for j in range(i+1,8):
                for k in range(8):
                    for m in range(k+1,8):
                        band1, band2, band3, band4 = str('y%d' % i), str('y%d' % j), str('y%d' % k), str('y%d' % m)
                        #print(i,j,k)
                        #if np.subtract(np.all(ph_data[band1]), np.all(ph_data[band2])) != np.subtract(np.all(ph_data[band3]), np.all(ph_data[band4])):
                        fig = plt.figure()
                        cmap = plt.get_cmap('gray_r')
                        counts,xbins,ybins = np.histogram2d(ph_data[band1] - ph_data[band2], ph_data[band3] - ph_data[band4],bins=30)
                        mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
                        mycolors = []
                        for l in range(len(mylevels)):
                            ival = l*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
                            mycolors.append(cmap(ival))
                        #reduce_C_function=np.sum,
                        
                        my_mean_func = lambda x: np.log10(np.mean(x))
                        my_sum_func  = lambda x: np.log10(np.sum(x))
                        
                        cax = plt.hexbin(ph_data[band1] - ph_data[band2], ph_data[band3] - ph_data[band4], C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-7.0, vmax=np.log10(0.3463695029598574))
                        
                        plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
                        plt.xlabel(str(magnitudes[i]) + "-" + str(magnitudes[j]))
                        plt.ylabel(str(magnitudes[k]) + "-" + str(magnitudes[m]))
                        
                        
                        cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%d}$"))
                        cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", rotation=270, labelpad = 15)
                        #file = str("/Users/Liana/Documents/PYTHON/Color_color/%s_%s_%s_%s.png" % (magnitudes[i], magnitudes[m], magnitudes[j],magnitudes[k]))
                        pdf.savefig()
                        plt.close()
    
    with PdfPages('COLOR_MAG.pdf') as pdf:
        #pdf = PdfPages('multipage_pdf.pdf')
        for i in range(8):
            for j in range(i+1,8):
                for k in range(8):
                    #for m in range(k+1,8):
                    band1, band2, band3 = str('y%d' % i), str('y%d' % j), str('y%d' % k)
                    #print(i,j,k)
                    #if np.subtract(np.all(ph_data[band1]), np.all(ph_data[band2])) != np.subtract(np.all(ph_data[band3]), np.all(ph_data[band4])):
                    fig = plt.figure()
                    cmap = plt.get_cmap('gray_r')
                    counts,xbins,ybins = np.histogram2d(ph_data[band3], ph_data[band1] - ph_data[band2], bins=30)
                    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
                    mycolors = []
                    for l in range(len(mylevels)):
                        ival = l*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
                        mycolors.append(cmap(ival))
                    #reduce_C_function=np.sum,
                    
                    my_mean_func = lambda x: np.log10(np.mean(x))
                    my_sum_func  = lambda x: np.log10(np.sum(x))
                    
                    cax = plt.hexbin(ph_data[band3], ph_data[band1] - ph_data[band2], C = data['y4'], gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis, reduce_C_function=my_mean_func, vmin=-7.0, vmax=np.log10(0.3463695029598574))
                    
                    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
                    plt.xlabel(str(magnitudes[k]))
                    plt.ylabel(str(magnitudes[i]) + "-" + str(magnitudes[j]))
                    
                    
                    cbar = plt.colorbar(cax, format=mtick.FormatStrFormatter(r"$10^{%d}$"))
                    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$", rotation=270, labelpad = 15)
                    #file = str("/Users/Liana/Documents/PYTHON/Color_color/%s_%s_%s_%s.png" % (magnitudes[i], magnitudes[m], magnitudes[j],magnitudes[k]))
                    pdf.savefig()
                    plt.close()
    #plt.show()
    
    hist, bin_edges = np.histogram(log_M, bins=100)
    #print(0.5*(bin_edges[1:] + bin_edges[:-1]))
    plt.figure()
    
    ax = axs[0]
    
    cax = ax.hexbin(log_M, log_SFR, C = data['y4'], bins = 'log' , gridsize=(20,20), mincnt = 1, cmap=plt.cm.viridis)
    ax.contour(counts.T, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    ax.set_xlabel("$\log(M_* [M_{\odot}])$")
    ax.set_ylabel("$\log(SFR [M_{\odot}/yr])$")
    cbar = fig.colorbar(cax, ax = ax)
    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$)", rotation=270, labelpad = 15)

    ax = axs[1]
    counts,xbins,ybins = np.histogram2d(log_M, log_sSFR,bins=30)
    mylevels = np.logspace(np.log10(1), np.log10(0.8*counts.max()), 5)
    mycolors = []
    for i in range(len(mylevels)):
        ival = i*((1.0 - 0.3)/(len(mylevels)-1.0)) + 0.3
        mycolors.append(cmap(ival))
    cax2 = ax.hexbin(log_M, log_sSFR, C = data['y4'], bins = 'log' , gridsize=(20,20),  mincnt = 1, cmap=plt.cm.viridis)
    ax.contour(counts.T, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    ax.set_xlabel("$\log(M_* [M_{\odot}])$")
    ax.set_ylabel("$\log(sSFR [yr^{-1}])$")
    #print(cax.get_array())
    #plt.vlines(20, 0, 4.5, 'r', '--')
    cbar2 = fig.colorbar(cax2, ax = ax)
    cbar2.set_label("Average $R_{GW, obs} [yr^{-1}]$)", rotation=270, labelpad = 15)

    plt.savefig('SFRvsM_subplots')
    plt.show()

    
    #norm=colors.LogNorm(vmin = 10**-6, vmax = 0.3)
    #mylevels = [10, 200, 500, 700, 10000]
    #cax = plt.hexbin(log_M, log_Z, C = R_obs_new, norm=colors.LogNorm(vmin = np.amin(R_obs_new), vmax = 1000), gridsize=(20,20), cmap=plt.cm.get_cmap('cubehelix', 6))
    cax = plt.hexbin(J_mag[index], u_mag[index] - g_mag[index], C = R_obs_all_data[index], gridsize=(20,20), bins = 'log', cmap=plt.cm.viridis)
    plt.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    plt.xlabel("J-Band")
    plt.ylabel("u-g")
    #plt.xlabel("$\log(Z)$")
    #plt.ylabel("$\log(SFR [M_{\odot}/yr])$")
    #plt.xlabel("$\log(M_* [M_{\odot}])$")
    #plt.ylabel("$\log(sSFR [yr^{-1}])$")
    cbar = fig.colorbar(cax)
    cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$)", rotation=270, labelpad = 15)
    plt.savefig('Jband_mag.png')
    plt.show()

    Chabrier_6 = np.load('Chabrier_z_6.npz')
    Chabrier_2 = np.load('Chabrier_z_2.npz')
    Chabrier_inst = np.load('Chabrier_instant.npz')
    
    print(RA, DEC)
    fig, ax = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    ax.scatter(DEC, RA, s =0.5)
    ax.set_xlabel("Dec")
    ax.set_ylabel("RA", labelpad=30)
    plt.show()
    
    fig, axs = plt.subplots(ncols=2, figsize=(10, 7))

    ax = axs[0]
    counts,xbins,ybins = np.histogram2d(log_M,log_SFR,bins=100)
    ax.hexbin(log_M, log_SFR, C = R_obs_all_data, gridsize=(20,20), bins = 'log', cmap=plt.cm.rainbow)
    ax.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    ax.set_xlabel("$\log(M_* [M_{\odot}])$")
    ax.set_ylabel("$\log(SFR [M_{\odot} /yr])$")
    #cbar = fig.colorbar(cax, ax = ax)
    #cbar.set_label("Average $R_{GW, obs} [yr^{-1}]$)", rotation=270, labelpad = 15)

    ax = axs[1]
    #ax = fig.add_axes([0,0,80,20])
    counts,xbins,ybins = np.histogram2d(log_M, log_sSFR, bins=100)
    cax2 = ax.hexbin(log_M, log_sSFR, C = R_obs_all_data, gridsize=(20,20), bins = 'log', cmap=plt.cm.rainbow)
    ax.contour(counts.T, levels=mylevels, colors=mycolors, extent=[xbins.min(), xbins.max(), ybins.min(), ybins.max()])
    ax.set_xlabel("$\log(M_* [M_{\odot}])$")
    ax.set_ylabel("$\log(sSFR [yr^{-1}])$")
    cbar2 = fig.colorbar(cax2, ax = ax)
    cbar2.set_label("Average $R_{GW, obs} [yr^{-1}]$)", rotation=270, labelpad = 15)
    plt.savefig('SFRvsM_subplots.png')
    plt.show()


    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    #R_plot = plt.scatter(lbt[0:len(all_MFH[0])], Chabrier_data['y1'][0:len(all_MFH[0])], c = all_MFH[0])
    #ax1.scatter(lbt[0:len(all_MFH[0])], Miller_data['y3'][0:len(all_MFH[0])], c = all_MFH[0])
    #ax1.scatter(lbt[0:len(all_MFH[0])], Kroupa_data['y3'][0:len(all_MFH[0])], c = all_MFH[0])
    #plt.scatter(lbt[0:len(all_MFH[0])], all_SFH[0], c = all_MFH[0])
    #cb = plt.colorbar(R_plot)
    #cb.set_label('Z', rotation=270, labelpad=10)
    #ax1.plot(lbt[0:len(all_MFH[0])], Chabrier_data['y1'][0:len(all_MFH[0])], '--', color="black", label = "Chabrier")
    ax1.plot(lbt[0:len(all_MFH[0])], Miller_data['y4'][0:len(all_MFH[0])]/Chabrier_data['y4'][0:len(all_MFH[0])], '--', color="grey", label = "Miller-Scalo/Chabrier")
    ax1.plot(lbt[0:len(all_MFH[0])], Kroupa_data['y4'][0:len(all_MFH[0])]/Chabrier_data['y4'][0:len(all_MFH[0])], '--', color="red", label = "Kroupa/Chabrier")
    #plt.plot(lbt[0:len(all_MFH[0])], all_SFH[0], color="grey", label = "SFR")
    ax1.set_xlabel("Lookback time $Gyr$")
    ax1.set_ylabel("$R_{GW}$ Ratio")
    ax1.legend(loc="center left")
    #ax1.set_xlim(11,13)
    #ax1.set_yscale('log')
    ax2 = ax1.twiny()
    ax2.set_xticks([lbt[0], lbt[100], lbt[120], lbt[140], lbt[160]])
    ax2.set_xticklabels([str(round(float(redshift[0]), 2)), str(round(float(redshift[100]), 2)), str(round(float(redshift[120]), 2)),str(round(float(redshift[140]), 2)), str(round(float(redshift[160]), 2))])
    ax2.set_xlabel("Redshift")
    plt.savefig('IMF_comparison_RGW_ratio_case2.png')
    #plt.yscale('log')
    plt.show()

    '''
