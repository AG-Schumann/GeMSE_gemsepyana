### Helper functions to do a simple analysis counting events in a fixed window
### - Sebastian Sep 2024

import numpy as np
import matplotlib.pyplot as plt

def count_line(gemsedata, thr_low, thr_high):
    _Xr = ((gemsedata.x>thr_low) & (gemsedata.x<thr_high))
    _nbins = np.count_nonzero(_Xr)
    _counts = gemsedata.y[_Xr].sum() 
    #print (f"Number of bins = {_nbins}")
    return _counts, _nbins

def cts_bgcorr(gd, thr):
    thr_sig = [thr[0], thr[1]]
    thr_bl = [thr[2], thr[3]]
    thr_bh = [thr[4], thr[5]]
    nsig, nb_sig = count_line(gemsedata=gd, thr_low=thr_sig[0], thr_high=thr_sig[1])
    nbgl, nb_l = count_line(gemsedata=gd, thr_low=thr_bl[0], thr_high=thr_bl[1])
    nbgh, nb_h = count_line(gemsedata=gd, thr_low=thr_bh[0], thr_high=thr_bh[1])
    ### we have to be careful here. the windows have the same size in energy, but not always contain the same amount of bins!
    nbg = (nbgl+nbgh)/(nb_l+nb_h) # average number of background counts per bin
    nsig_corr = nsig-(nbg*nb_sig)
    #print (f"nsig={nsig}, nbgl={nbgl}, nbgh={nbgh}, nsig_corr={nsig_corr}")
    ## and the errors:
    nsig_e = np.sqrt(nsig)
    nbgl_e = np.sqrt(nbgl)
    nbgh_e = np.sqrt(nbgh)
    #nbg_e = np.sqrt( (nbgl_e/2.)**2 + (nbgh_e/2.)**2 )    
    nbg_e = np.sqrt( (nbgl_e)**2 + (nbgh_e)**2 ) / (nb_l+nb_h) * nb_sig # error on average number of background counts per bin TIMES number of bins in signal region

    nsig_corr_e = np.sqrt((nsig_e)**2 + (nbg_e)**2)
    #if detailed_results:
    return (nsig_corr, nsig_corr_e, nsig, nbgl, nbgh)
    #return (nsig_corr, nsig_corr_e)

def compute_activity_per_line(gd, thr, eff, mass=1):
    _cts, _cts_e, nsig, nbgl, nbgh = cts_bgcorr(gd=gd, thr=thr)
    _t = gd.t_live
    _a = _cts/_t/eff[0]/mass
    _a_e = np.sqrt( (_a * np.sqrt((_cts_e/_cts)**2 + (eff[1]/eff[0])**2))**2)
    return np.array([_a, _a_e, nsig, nbgl, nbgh])



def convert_isotope_name(isotope):
    import re
    # Use regex to capture the letters and digits
    match1 = re.match(r"([A-Za-z]+)(\d+)", isotope)
    match2 = re.match(r"(\d+)([A-Za-z]+)", isotope)
    if match1:
        # Reorder to place digits first followed by the element symbol
        element = match1.group(1)
        mass_number = match1.group(2)
        return f"{mass_number}{element}"
    elif match2:
        element = match2.group(2)
        mass_number = match2.group(1)
        return f"{element}{mass_number}"
    else:
        raise ValueError("Input isotope format is invalid")



def simple_activities(gd, plot=False, _xw=3, isotopes=None, yscale='log'):
    nplots = 0
    actvs = []
    als = {}
    for ke in gd.eff_dict:
        if isotopes:
            if not ke in isotopes:
                continue
        if ke in gd.iso_dict:
            print ("---------")
            #print (gd.iso_dict[ke]['Peak Energies (keV)'])
            #continue
            for i in range(len(gd.iso_dict[ke]['Peak Energies (keV)'])):
                #print (ke, enrg)
                enrg = gd.iso_dict[ke]['Peak Energies (keV)'][i]
                lfr = gd.iso_dict[ke]['Lower Fit Range (keV)'][i]
                ufr = gd.iso_dict[ke]['Upper Fit Range (keV)'][i]
                de = ufr-lfr
                bl = lfr-de
                bh = ufr+de
                _thr = [lfr,ufr,bl,lfr,ufr,bh]
                if enrg in gd.eff_dict[ke]:
                    eff = gd.eff_dict[ke][enrg]
                    a=compute_activity_per_line(gd=gd, thr=_thr, eff=eff ) # returns acitivity in Bq (not mBq!)
                    actvs.append(a)
                    if not ke in als:
                        als[ke] = {"enrg_lab":[], "activity":[], "activity_unc":[]}
                    als[ke]["activity"].append(a[0])
                    als[ke]["activity_unc"].append(a[1])
                    als[ke]["enrg_lab"].append(f"{enrg} keV")
                    #print (f"{ke}: {enrg} keV; eff=({eff[0]:.1e}+-{eff[1]:.1e});thr={_thr} -> ({a[0]*1e3:.1f}+-{a[1]*1e3:.1f}) mBq")
                    print (f"{ke}: {enrg} keV; eff=({eff[0]:.1e}+-{eff[1]:.1e}); (a2={a[2]}, a3={a[3]}, a4={a[4]}) -> ({a[0]*1e3:.1f}+-{a[1]*1e3:.1f}) mBq")
                    nplots +=1
                else:
                    eff = (0,0)
                    print (f"No eff for {ke}: {enrg} keV. Skipping line.")
    for ke in als:
        als[ke]["activity"] = np.array( als[ke]["activity"] )
        als[ke]["activity_unc"] = np.array( als[ke]["activity_unc"] )
        als[ke]["enrg_lab"] = np.array( als[ke]["enrg_lab"] )

    actvs = np.array(actvs)
                
    if plot:
        plt.style.use('/home/sebastian/.pltstyle/gemse_small.mplstyle')
        fig, axs = plt.subplots(nplots,1,figsize=(16,nplots*8))
        na = -1
        for ke in gd.eff_dict:
            if isotopes:
                if not ke in isotopes:
                    continue
            if ke in gd.iso_dict:
                for i in range(len(gd.iso_dict[ke]['Peak Energies (keV)'])):
                    enrg = gd.iso_dict[ke]['Peak Energies (keV)'][i]
                    if not enrg in gd.eff_dict[ke]:
                        print (f"Attention! {enrg} not in eff_dict!")
                        continue
                    na += 1
                    if nplots>1:
                        ax = axs[na]
                    else:
                        ax=axs
                    lfr = gd.iso_dict[ke]['Lower Fit Range (keV)'][i]
                    ufr = gd.iso_dict[ke]['Upper Fit Range (keV)'][i]
                    de = ufr-lfr
                    bl = lfr-de
                    bh = ufr+de
                    eff = gd.eff_dict[ke][enrg]

                    _thr = [lfr,ufr,bl,lfr,ufr,bh]
                    for th in _thr:
                        ax.axvline(x=th, ls='--', color='k', lw=1)
                    ax.axvline(x=enrg, ls='-', color='r', lw=1)

                    ax.hlines(y=actvs[na][3]/np.count_nonzero( (gd.x>_thr[2])&(gd.x<_thr[3]) ), xmin=_thr[2], xmax=_thr[3], ls='-', color='r', lw=1)
                    ax.hlines(y=actvs[na][4]//np.count_nonzero( (gd.x>_thr[4])&(gd.x<_thr[5]) ), xmin=_thr[4], xmax=_thr[5], ls='-', color='r', lw=1)                    

                    if actvs[na][0]*1e3 > 0.1:
                        _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1f}+-{actvs[na][1]*1e3:.1f}) mBq"
                    else:
                        _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1e}+-{actvs[na][1]*1e3:.1e})mBq"
                    _xr = ((gd.x>(_thr[2]-_xw*de))&(gd.x<(_thr[5]+_xw*de)))
                    
                    ax.plot(gd.x[_xr] , gd.y[_xr] , '-o', c='b', lw=1, label=_lab)

                    if True:
                        #xrange = (gd.x>_thr[0])&(gd.x<_thr[1]) # draw special lines onlz in ROI window 
                        xrange = (_thr[2],_thr[5])# draw special lines onlz in ROI window 
                        gd.draw_special_lines(sdict = gd.manual_dict, ## put cuts on manual dict, to select important lines
                              col = 'green',
                              ax = ax,
                              xrange=xrange, 
                              minBR = 0,
                              isotope= None,
                             ) 
                    
                    ax.set_ylabel('counts per bin [1]')
                    ax.set_yscale(yscale)
                    ax.set_xlabel('energy [keV]')
                    #ax.set_xlim(bl-de, bh+de)
                    ax.legend(title=f"{gd.sample_name}")
        fig.show()
    return actvs, als # ,np.array(als)


def plot_actvs(als):
    ## This function is used to plot the results of multiple gamma lines that belong to a single isotope in the same figure
    ## Input is the result of the simple_activities function 
    nisos = len(als.keys())
    fig, axs = plt.subplots(nisos,1,figsize=(12,nisos*6))
    plt.style.use('/home/sebastian/.pltstyle/gemse.mplstyle')
    for i, ke in enumerate( als.keys() ):
        if nisos > 1:
            ax = axs[i]
        else:
            ax = axs
        XnegA = (als[ke]['activity'] > 0)

        ax.errorbar(x=als[ke]['enrg_lab'][XnegA], y=als[ke]['activity'][XnegA], yerr=als[ke]['activity_unc'][XnegA], fmt='o', label=f"{ke}")

        try:
            #weight = 1/(uncertainty)^2
            avg = np.average(als[ke]['activity'][XnegA], weights=1/(als[ke]['activity_unc'][XnegA])**2)
            ax.axhline(y=avg, ls='--', color='r', lw=1, label=f"avg (weighted): {avg*1000:.1f} mBq")
        except:
            pass
        
        ax.legend() 
        ax.set_ylabel("Activity [Bq]")
    fig.show()


def which_chain(gd, isotope):
    _chains = []
    for k in gd.decay_chains.keys():
        if isotope in gd.decay_chains[k] or convert_isotope_name(isotope=isotope) in gd.decay_chains[k]:
            _chains.append(k)
    print (_chains)


