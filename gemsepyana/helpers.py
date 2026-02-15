### Helper functions to do a simple analysis counting events in a fixed window
### - Sebastian Sep 2024

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from uncertainties import ufloat

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

def compute_activity_per_line(gd, thr, eff, mass=1, bg=None):
    _cts, _cts_e, nsig, nbgl, nbgh = cts_bgcorr(gd=gd, thr=thr)
    _t = gd.t_live
    _a = _cts/_t/eff[0]/mass
    _a_e = np.sqrt( (_a * np.sqrt((_cts_e/_cts)**2 + (eff[1]/eff[0])**2))**2 ) # outer np.sqrt( ()**2) to avoid negative uncertainties

    if bg:
      _cts_bg, _cts_e_bg, nsig_bg, nbgl_bg, nbgh_bg = cts_bgcorr(gd=bg, thr=thr)
      _t_bg = bg.t_live
      _a_bg = _cts_bg/_t_bg
      _a_e_bg = np.sqrt( ( _a_bg * _cts_e_bg/_cts_bg )**2 )

      a = (_a*mass - _a_bg)/mass
      a_e = np.sqrt( (_a_e)**2 + (_a_e_bg)**2 )
      #return np.array(a, a_e, _a_bg, _a_e_bg)
      return {"signal": np.array([_a, _a_e, nsig, nbgl, nbgh, _cts, _cts_e]), "signal_corr": np.array([a, a_e]), "background": np.array([_a_bg, _a_e_bg, nsig_bg, nbgl_bg, nbgh_bg, _cts_bg, _cts_e_bg])}
    else:
      #return np.array([_a, _a_e, nsig, nbgl, nbgh])
      #return np.array([_a, _a_e, _cts, _cts_e, nsig, nbgl, nbgh]), None
      return {"signal": np.array([_a, _a_e, nsig, nbgl, nbgh, _cts, _cts_e]), "background": None, "signal_corr": None}



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



def simple_activities(gd, plot=False, _xw=3, isotopes=None, yscale='log', bg=None):
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
                    capl=compute_activity_per_line(gd=gd, thr=_thr, eff=eff, bg=bg ) # returns acitivity in Bq (not mBq!)
                      # returns np.array with: ([_a, _a_e, nsig, nbgl, nbgh, _cts, _cts_e])
                    a = capl["signal"]
                    acorr = capl["signal_corr"]
                    b = capl["background"]
                    actvs.append(a)
                    if not ke in als:
                        als[ke] = {"enrg_lab":[], "activity":[], "activity_unc":[], "activity_bg_corrected":[], "activity_bg_corrected_unc":[],  "activity_unc_inclSysUnc":[], "actvs":{} }
                    als[ke]["actvs"][enrg]=capl
                    als[ke]["activity"].append(a[0])
                    als[ke]["activity_unc"].append(a[1]) # this is the stats uncertainty due to the number of observed counts only
                    #_nstat_mcstat_unc = (a[0]*np.sqrt( (a[1]/a[0])**2 + (eff[1]/eff[0])**2  )) # adding rel. error coming from finite MC stas (in quadrature)
                    #als[ke]["activity_unc_inclStatUncSim"].append( _nstat_mcstat_unc  )
                    _nstat_mcstat_mcsys = a[0]*np.sqrt( (a[1]/a[0])**2 + (0.1)**2 ) # adding 10% systematic error on MC efficiency
                    als[ke]["activity_unc_inclSysUnc"].append( _nstat_mcstat_mcsys  )
                    als[ke]["activity_bg_corrected"].append(acorr[0])
                    als[ke]["activity_bg_corrected_unc"].append(acorr[1])
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
        als[ke]["activity_unc_inclSysUnc"] = np.array( als[ke]["activity_unc_inclSysUnc"] )
        als[ke]["activity_bg_corrected"] = np.array( als[ke]["activity_bg_corrected"] )
        als[ke]["activity_bg_corrected_unc"] = np.array( als[ke]["activity_bg_corrected_unc"] )
        als[ke]["enrg_lab"] = np.array( als[ke]["enrg_lab"] )

    actvs = np.array(actvs)
                
    if plot:
        plt.style.use( os.path.join( os.path.dirname(__file__), 'gemse.mplstyle') )
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

                    ax.hlines(y=actvs[na][3]/np.count_nonzero( (gd.x>_thr[2])&(gd.x<_thr[3]) ), xmin=_thr[2], xmax=_thr[3], ls='-', color='r', lw=1, zorder=25)
                    ax.hlines(y=actvs[na][4]//np.count_nonzero( (gd.x>_thr[4])&(gd.x<_thr[5]) ), xmin=_thr[4], xmax=_thr[5], ls='-', color='r', lw=1, zorder=26)                    

                    if actvs[na][0]*1e3 > 0.1:
                        _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1f}+-{actvs[na][1]*1e3:.1f}) mBq"
                    else:
                        _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1e}+-{actvs[na][1]*1e3:.1e})mBq"
                    _xr = ((gd.x>(_thr[2]-_xw*de))&(gd.x<(_thr[5]+_xw*de)))
                    _xr_bg = ((bg.x>(_thr[2]-_xw*de))&(bg.x<(_thr[5]+_xw*de)))
                    
                    ax.plot(gd.x[_xr] , gd.y[_xr] , '-o', c='b', lw=1, label=_lab)
                    if bg:
                      ax.plot(bg.x[_xr_bg] , bg.y[_xr_bg] , '-o', c='g', lw=1, label="background")


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


def plot_actvs(als, exclude=None, ignore_outliers=True):
    ## This function is used to plot the results of multiple gamma lines that belong to a single isotope in the same figure
    ## Input is the result of the simple_activities function 
    nisos = len(als.keys())
    fig, axs = plt.subplots(nisos,1,figsize=(12,nisos*6))
#    plt.style.use('/home/sebastian/.pltstyle/gemse.mplstyle')
    plt.style.use( os.path.join( os.path.dirname(__file__), 'gemse.mplstyle') )

    for i, ke in enumerate( als.keys() ):
        if nisos > 1:
            ax = axs[i]
        else:
            ax = axs

        _xid = np.array( len(als[ke]['activity'])*[ True ] )

        if not exclude is None and ke in exclude.keys():
          for dx in exclude[ke]:
              if dx < len(_xid):
                  _xid[dx] = False

        try:
          XnegA = (als[ke]['activity'] > 0) & _xid
        except Exception as e:
          print (f"...Error constructing cut XnegA. {e=}")
          pass


        if ignore_outliers:
          # Plot all data to set proper x-range (but suppress visibility)
          ax.plot(als[ke]['enrg_lab'][XnegA], als[ke]['activity'][XnegA], visible=False, zorder=1)
        else:
          ax.plot(als[ke]['enrg_lab'], als[ke]['activity'], visible=False, zorder=1)

        try:
            #weight = 1/(uncertainty)^2
            _weights = 1 / (als[ke]['activity_unc'][XnegA])**2
            avg = np.average(als[ke]['activity'][XnegA], weights=_weights)
            avg_unc = np.sqrt( 1 / np.sum(_weights) )
            avg_unc_inclSys = avg * np.sqrt((avg_unc/avg)**2 +(0.1)**2)
            uavg = ufloat(avg, avg_unc)
            uavg_inclSys = ufloat(avg, avg_unc_inclSys)
            #label = f"avg (weighted): ({avg*1000:.1f}+-{avg_unc*1000:.1f}) mBq"
            ulabel = f"avg (weighted): ({uavg} / {uavg_inclSys}) mBq"
            ax.axhline(y=avg, ls='--', color='blue', lw=1, label=ulabel)

            # Get the x and y limits of the axis
            x_min, x_max = ax.get_xlim()
            y_min = avg-avg_unc
            y_max = avg+avg_unc
            y_minSys = avg-avg_unc_inclSys
            y_maxSys = avg+avg_unc_inclSys
            #print (f"{x_min=} {x_max=} {y_min=} {y_max=}")

            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     color='lightblue', alpha=0.5, zorder=2)

            rectSys = patches.Rectangle((x_min, y_minSys), x_max - x_min, y_maxSys - y_minSys, 
                                     color='lightgreen', alpha=0.4, zorder=1)
            ax.add_patch(rect)
            ax.add_patch(rectSys)
        except Exception as e:
          print (f"ERROR {e=}")
          pass

        #higher zorder are drawn on top
        ax.errorbar(x=als[ke]['enrg_lab'][XnegA], y=als[ke]['activity'][XnegA], yerr=als[ke]['activity_unc'][XnegA], color='k', fmt='o', label=f"{ke}", zorder=12)
        ax.errorbar(x=als[ke]['enrg_lab'][XnegA], y=als[ke]['activity'][XnegA], yerr=als[ke]['activity_unc_inclSysUnc'][XnegA], color='g', fmt='o', label="includig stat. and 10% sys. unc. of MC", zorder=9)
        ax.errorbar(x=als[ke]['enrg_lab'][XnegA], y=als[ke]['activity_bg_corrected'][XnegA], yerr=als[ke]['activity_bg_corrected_unc'][XnegA], color='m', fmt='o', label="bg corrected", zorder=15)
        if not exclude is None and not ignore_outliers:
          ax.errorbar(x=als[ke]['enrg_lab'][~XnegA], y=als[ke]['activity'][~XnegA], yerr=als[ke]['activity_unc'][~XnegA], color='k', alpha=0.4, fmt='o', zorder=11)

        #eff = gd.eff_dict[ke][enrg]
        
        ax.legend() 
        ax.set_ylabel("Activity [Bq]")
    fig.show()


def which_chain(gd, isotope):
    _chains = []
    for k in gd.decay_chains.keys():
        if isotope in gd.decay_chains[k] or convert_isotope_name(isotope=isotope) in gd.decay_chains[k]:
            _chains.append(k)
    print (_chains)


def print_activities_summary(als, exclude=None, ignore_outliers=True):
    for i, ke in enumerate( als.keys() ):
        _xid = np.array( len(als[ke]['activity'])*[ True ] ) 
        if not exclude is None and ke in exclude.keys():
          for dx in exclude[ke]:
              if dx < len(_xid):
                  _xid[dx] = False
        try:
          XnegA = (als[ke]['activity'] > 0) & _xid
        except Exception as e:
          print (f"...Error constructing cut XnegA. {e=}")
          pass

        try:
          is_bg_corrected=False
          if not als[ke]['activity_bg_corrected'].any() is None:
            _weights = 1 / (als[ke]['activity_bg_corrected_unc'][XnegA])**2
            avg = np.average(als[ke]['activity_bg_corrected'][XnegA], weights=_weights)
            is_bg_corrected=True
          else:
            _weights = 1 / (als[ke]['activity_unc'][XnegA])**2
            avg = np.average(als[ke]['activity'][XnegA], weights=_weights)
          avg_unc = np.sqrt( 1 / np.sum(_weights) )
          avg_unc_inclSys = avg * np.sqrt((avg_unc/avg)**2 +(0.1)**2)
          uavg = ufloat(avg, avg_unc)
          uavg_inclSys = ufloat(avg, avg_unc_inclSys)
          print (f"{ke:>10}:  {uavg:S} Bq \t-> {uavg_inclSys:S} Bq; \t{is_bg_corrected=}")
        except Exception as e:
            print (f"{ke:>10}: {e}")


