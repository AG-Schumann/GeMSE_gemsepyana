### Helper functions to do a simple analysis counting events in a fixed window
### - Sebastian Sep 2024

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from uncertainties import ufloat
import uncertainties

def result_or_simple_upper_limit(v, v_unc=None, k=2):
  if v_unc is None:
    assert type(v) == uncertainties.core.Variable
    _v = v
  else:
    _v = ufloat(v, v_unc)

  if _v.std_score(0) > -k: # std_score checks if the value _v is larger than zero by at least k * sigma
    ## if True, the value is larger than zero by less than k*uncertainty
    ## Hence, report (simple) UL!
    return np.abs(2*_v.std_dev)
  else:
    ## We can report an actual number:
    return _v

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
    ###############################################################################################
    ## modification March 2025; use only left OR right region to assess flat background component:
    nsig_corr_left = nsig-(nbgl/nb_l)*nb_sig
    nsig_corr_right = nsig-(nbgh/nb_h)*nb_sig
    ###############################################################################################
    #print (f"nsig={nsig}, nbgl={nbgl}, nbgh={nbgh}, nsig_corr={nsig_corr}")
    ## and the errors:
    nsig_e = np.sqrt(nsig)
    nbgl_e = np.sqrt(nbgl)
    nbgh_e = np.sqrt(nbgh)
    #nbg_e = np.sqrt( (nbgl_e/2.)**2 + (nbgh_e/2.)**2 )    
    nbg_e = np.sqrt( (nbgl_e)**2 + (nbgh_e)**2 ) / (nb_l+nb_h) * nb_sig # error on average number of background counts per bin TIMES number of bins in signal region

    nsig_corr_e = np.sqrt((nsig_e)**2 + (nbg_e)**2)
    #if detailed_results:
    return (nsig_corr, nsig_corr_e, nsig, nbgl, nbgh, nsig_corr_left, nsig_corr_right)
    #return (nsig_corr, nsig_corr_e)

def compute_activity_per_line(gd, thr, eff, mass=1, bg=None):
    _cts, _cts_e, nsig, nbgl, nbgh, nsig_corr_left, nsig_corr_right = cts_bgcorr(gd=gd, thr=thr)
    _t = gd.t_live
    _a = _cts/_t/eff[0]/mass
    ########## mod March 2025 ############
    _a_left = nsig_corr_left/_t/eff[0]/mass
    _a_right = nsig_corr_right/_t/eff[0]/mass
    ######################################
    _a_e = np.sqrt( (_a * np.sqrt((_cts_e/_cts)**2 + (eff[1]/eff[0])**2))**2 ) # outer np.sqrt( ()**2) to avoid negative uncertainties

    if bg:
      _cts_bg, _cts_e_bg, nsig_bg, nbgl_bg, nbgh_bg, nsig_corr_left_bg, nsig_corr_right_bg = cts_bgcorr(gd=bg, thr=thr)
      _t_bg = bg.t_live
      _a_bg = _cts_bg/_t_bg
      _a_e_bg = np.sqrt( ( _a_bg * _cts_e_bg/_cts_bg )**2 )

      a = (_a*mass - _a_bg)/mass
      a_e = np.sqrt( (_a_e)**2 + (_a_e_bg)**2 )
      ########## mod March 2025 ############
      a_left = (_a_left*mass - _a_bg)/mass
      a_right = (_a_right*mass - _a_bg)/mass
      ######################################
      #return np.array(a, a_e, _a_bg, _a_e_bg)
      return {"signal": np.array([_a, _a_e, nsig, nbgl, nbgh, _cts, _cts_e]), "signal_corr": np.array([a, a_e]), "background": np.array([_a_bg, _a_e_bg, nsig_bg, nbgl_bg, nbgh_bg, _cts_bg, _cts_e_bg]), "signal_left": np.array([a_left, a_e]), "signal_right": np.array([a_right, a_e])}
    else:
      #return np.array([_a, _a_e, nsig, nbgl, nbgh])
      #return np.array([_a, _a_e, _cts, _cts_e, nsig, nbgl, nbgh]), None
      return {"signal": np.array([_a, _a_e, nsig, nbgl, nbgh, _cts, _cts_e]), "background": None, "signal_corr": None, "signal_left": np.array([_a_left, _a_e]), "signal_right": np.array([_a_right, _a_e])}



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



def simple_activities(gd, plot=False, _xw=3, isotopes=None, yscale='log', bg=None, show_summary=False, ignore_outliers=True):
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
                      als[ke] = {"enrg_lab":[], "activity":[], "activity_unc":[], "activity_bg_corrected":[], "activity_bg_corrected_unc":[],  "activity_unc_inclSysUnc":[], "actvs":{}, "activity_left":[], "activity_right":[] }
                    als[ke]["actvs"][enrg]=capl
                    als[ke]["activity"].append(a[0])
                    als[ke]["activity_left"].append(capl["signal_left"][0])
                    als[ke]["activity_right"].append(capl["signal_right"][0])
                    als[ke]["activity_unc"].append(a[1]) # this is the stats uncertainty due to the number of observed counts only
                    #_nstat_mcstat_unc = (a[0]*np.sqrt( (a[1]/a[0])**2 + (eff[1]/eff[0])**2  )) # adding rel. error coming from finite MC stas (in quadrature)
                    #als[ke]["activity_unc_inclStatUncSim"].append( _nstat_mcstat_unc  )
                    _nstat_mcstat_mcsys = a[0]*np.sqrt( (a[1]/a[0])**2 + (0.1)**2 ) # adding 10% systematic error on MC efficiency
                    als[ke]["activity_unc_inclSysUnc"].append( _nstat_mcstat_mcsys  )
                    als[ke]["activity_bg_corrected"].append(acorr[0])
                    als[ke]["activity_bg_corrected_unc"].append(acorr[1])
                    als[ke]["enrg_lab"].append(f"{enrg} keV")
                    #print (f"{ke}: {enrg} keV; eff=({eff[0]:.1e}+-{eff[1]:.1e});thr={_thr} -> ({a[0]*1e3:.1f}+-{a[1]*1e3:.1f}) mBq")
                    #print (f"{ke}: {enrg} keV; eff=({eff[0]:.1e}+-{eff[1]:.1e}); (a2={a[2]}, a3={a[3]}, a4={a[4]}) -> ({a[0]*1e3:.1f}+-{a[1]*1e3:.1f}) mBq")
                    ueff = ufloat(eff[0],eff[1])
                    ua= ufloat(a[0],a[1])
                    ual= ufloat(capl["signal_left"][0],a[1])
                    uar= ufloat(capl["signal_right"][0],a[1])
                    print (f"{ke}: {enrg} keV; eff={ueff:%S}; a={ua:S}Bq (left:{ual:S}Bq; right:{uar:S}Bq)")
                    nplots +=1
                else:
                    eff = (0,0)
                    print (f"No eff for {ke}: {enrg} keV. Skipping line.")
    for ke in als:
        als[ke]["activity"] = np.array( als[ke]["activity"] )
        als[ke]["activity_left"] = np.array( als[ke]["activity_left"] )
        als[ke]["activity_right"] = np.array( als[ke]["activity_right"] )
        als[ke]["activity_unc"] = np.abs( np.array( als[ke]["activity_unc"] ) )
        als[ke]["activity_unc_inclSysUnc"] = np.abs( np.array( als[ke]["activity_unc_inclSysUnc"] ) )
        als[ke]["activity_bg_corrected"] = np.array( als[ke]["activity_bg_corrected"] )
        als[ke]["activity_bg_corrected_unc"] = np.abs( np.array( als[ke]["activity_bg_corrected_unc"] ) )
        als[ke]["enrg_lab"] = np.array( als[ke]["enrg_lab"] )

    actvs = np.array(actvs)
                
    if plot:
        #plt.style.use('/home/sebastian/.pltstyle/gemse_small.mplstyle')
        plt.style.use( os.path.join( os.path.dirname(__file__), 'gemse.mplstyle') )
        if show_summary:
          nplots = nplots + len(isotopes)
        fig, axs = plt.subplots(nplots,1,figsize=(16,nplots*8))
        na = -1
        naa = -1
        for ke in gd.eff_dict:
            if isotopes:
                if not ke in isotopes:
                    continue
            if ke in gd.iso_dict:
                jk = -1
                for i in range(len(gd.iso_dict[ke]['Peak Energies (keV)'])):
                    enrg = gd.iso_dict[ke]['Peak Energies (keV)'][i]
                    if not enrg in gd.eff_dict[ke]:
                        print (f"Attention! {enrg} not in eff_dict!")
                        continue
                    na += 1
                    naa += 1
                    jk += 1
                    if nplots>1:
                        ax = axs[naa]
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

                    #if actvs[na][0]*1e3 > 0.1:
                    #    _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1f}+-{actvs[na][1]*1e3:.1f}) mBq"
                    #else:
                    #    _lab = f"{ke}, {enrg} keV, eff_BR=({eff[0]*100:.2f}+-{eff[1]*100:.2f})%\n({actvs[na][0]*1e3:.1e}+-{actvs[na][1]*1e3:.1e})mBq"

                    uaavg=ufloat(actvs[na][0], actvs[na][1])
                    ueff=ufloat(eff[0],eff[1])
                    #_lab = f"{ke}, {enrg}keV, eff_BR={ueff:%S}, {uaavg:S}Bq"
                    _lab = f"eff_BR={ueff:%S}, {uaavg:S}Bq"

                    #### add left/right bg subtraction results to legend
                    #### use dirty trick of an empty scatter plot to place the labels
                    _uaa = ufloat(als[ke]['activity'][jk], actvs[na][1])
                    _ual = ufloat(als[ke]['activity_left'][jk], actvs[na][1])
                    _uar = ufloat(als[ke]['activity_right'][jk], actvs[na][1])

                    ax.scatter([], [], color="w", alpha=0, label=f"a_mean ={_uaa:S}Bq")
                    ax.scatter([], [], color="w", alpha=0, label=f"a_left ={_ual:S}Bq")
                    ax.scatter([], [], color="w", alpha=0, label=f"a_right={_uar:S}Bq")
                    #print (f"{als[ke]['activity_left'][jk]=}")
                    ##
                    _xr = ((gd.x>(_thr[2]-_xw*de))&(gd.x<(_thr[5]+_xw*de)))
                    _xr_bg = ((bg.x>(_thr[2]-_xw*de))&(bg.x<(_thr[5]+_xw*de)))
                    
                    ax.plot(gd.x[_xr] , gd.y[_xr] , '-o', c='b', lw=1, label=_lab)
                    if bg:
                      ax.plot(bg.x[_xr_bg] , bg.y[_xr_bg] , '-o', c='g', lw=1, label="background")
                    ax.text(0.05, 0.95, f'{ke} ({enrg}keV)', transform=ax.transAxes, fontsize=24, verticalalignment='top', horizontalalignment='left',zorder=200,  bbox=dict(facecolor='white', edgecolor='white', boxstyle='round,pad=0.3'))


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
                    ax.legend(title=f"{gd.sample_name}", loc="upper right")
                if show_summary:
                    _als = {}
                    _als[ke]=als[ke]
                    naa+=1
                    plot_actvs(als=_als, ax=axs[naa], ignore_outliers=ignore_outliers)

        fig.show()
    return actvs, als # ,np.array(als)


def plot_actvs(als, exclude=None, ignore_outliers=True, axs=None):
    ## This function is used to plot the results of multiple gamma lines that belong to a single isotope in the same figure
    ## Input is the result of the simple_activities function 
    nisos = len(als.keys())
    if axs is None:
      fig, axs = plt.subplots(nisos,1,figsize=(12,nisos*6))
    else:
      fig = plt.gcf()
      
#    plt.style.use('/home/sebastian/.pltstyle/gemse.mplstyle')
    plt.style.use( os.path.join( os.path.dirname(__file__), 'gemse.mplstyle') )

    for i, ke in enumerate( als.keys() ):
        if not fig is None:
          if nisos > 1:
              ax = axs[i]
          else:
              ax = axs

        _xid = np.array( len(als[ke]['activity'])*[ True ] )

        if not exclude is None and ke in exclude.keys():
          for dx in exclude[ke]:
              if dx < len(_xid):
                  _xid[dx] = False

        #try:
        #  XnegA = (als[ke]['activity'] > 0) & _xid
        #except Exception as e:
        #  print (f"...Error constructing cut XnegA. {e=}")
        #  pass


        if ignore_outliers:
          XnegA = (als[ke]['activity'] > 0) & _xid
          # Plot all data to set proper x-range (but suppress visibility)
          ax.plot(als[ke]['enrg_lab'][XnegA], als[ke]['activity'][XnegA], visible=False, zorder=1)
        else:
          XnegA = _xid
          ax.plot(als[ke]['enrg_lab'], als[ke]['activity'], visible=False, zorder=1)

        of=0.05
        try:
            #weight = 1/(uncertainty)^2
            _weights = 1 / (als[ke]['activity_unc'][XnegA])**2
            avg = np.average(als[ke]['activity'][XnegA], weights=_weights)
            avg_unc = np.sqrt( 1 / np.sum(_weights) )
            avg_unc_inclSys = avg * np.sqrt((avg_unc/avg)**2 +(0.1)**2)
            uavg = ufloat(avg, avg_unc)
            uavg_inclSys = ufloat(avg, avg_unc_inclSys)
            #label = f"avg (weighted): ({avg*1000:.1f}+-{avg_unc*1000:.1f}) mBq"
            #ulabel = f"avg (weighted): ({uavg} / {uavg_inclSys}) mBq"
            ulabel = f"weighted mean: {uavg} mBq; incl 10% sys: {uavg_inclSys} mBq"
            ax.axhline(y=avg, ls='--', color='blue', lw=1, label=ulabel)

            # Get the x and y limits of the axis
            x_min, x_max = ax.get_xlim()
            y_min = avg-avg_unc
            y_max = avg+avg_unc
            y_minSys = avg-avg_unc_inclSys
            y_maxSys = avg+avg_unc_inclSys
            #print (f"{x_min=} {x_max=} {y_min=} {y_max=}")

            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, 
                                     color='lightblue', alpha=0.5, zorder=5)
            ax.text(x_min+of, y_max-0.1*(y_max-y_min), f"({uavg}) mBq", color="white", alpha=0.7, zorder=6, ha="left", va="top", size=16)

            rectSys = patches.Rectangle((x_min, y_minSys), x_max - x_min, y_maxSys - y_minSys, 
                                     color='lightgreen', alpha=0.4, zorder=1)
            ax.text(x_min+of, y_maxSys-0.1*(y_maxSys-y_minSys), f"({uavg_inclSys}) mBq", color="white", alpha=0.7, zorder=2, ha="left", va="top", size=16)

            ax.add_patch(rect)
            ax.add_patch(rectSys)
            ax.set_xlim((x_min, x_max))
        except Exception as e:
          print (f"ERROR {e=}")
          pass

        # shift errorbar plots slightly along x:
        xpos = np.arange(len( als[ke]['enrg_lab'][XnegA] ))
        #higher zorder are drawn on top
        ax.errorbar(x=xpos-2*of, y=als[ke]['activity_left'][XnegA], yerr=als[ke]['activity_unc'][XnegA], color='lightgray', mfc="white", fmt='<', label=f"{ke}_left", zorder=12)
        ax.errorbar(x=xpos-1*of, y=als[ke]['activity'][XnegA], yerr=als[ke]['activity_unc'][XnegA], color='lightblue', fmt='o', label=f"{ke}", zorder=12)
        ax.errorbar(x=xpos+0*of, y=als[ke]['activity_right'][XnegA], yerr=als[ke]['activity_unc'][XnegA], color='lightgray', mfc="white", fmt='>', label=f"{ke}_right", zorder=12)
        ax.errorbar(x=xpos+1*of, y=als[ke]['activity_bg_corrected'][XnegA], yerr=als[ke]['activity_bg_corrected_unc'][XnegA], color='darkgray', fmt='o', label="bg corrected", zorder=15)
        ax.errorbar(x=xpos+2*of, y=als[ke]['activity'][XnegA], yerr=als[ke]['activity_unc_inclSysUnc'][XnegA], color='black', fmt='o', label="includig stat. and 10% sys. unc. of MC", zorder=19)
        if not exclude is None and not ignore_outliers:
          ax.errorbar(x=xpos+3*of, y=als[ke]['activity'][~XnegA], yerr=als[ke]['activity_unc'][~XnegA], color='red', alpha=0.4, fmt='o', zorder=21)

        #eff = gd.eff_dict[ke][enrg]
        ax.set_xticks(xpos, als[ke]['enrg_lab'][XnegA])
        ax.legend() 
        ax.set_ylabel("Activity [Bq]")
    if not fig is None:
      fig.show()


def which_chain(gd, isotope):
    _chains = []
    for k in gd.decay_chains.keys():
        if isotope in gd.decay_chains[k] or convert_isotope_name(isotope=isotope) in gd.decay_chains[k]:
            _chains.append(k)
    print (_chains)


def print_activities_summary(als, exclude=None, ignore_outliers=True, mass=None, pcs=None):
    res_dict = {}
    for i, ke in enumerate( als.keys() ):
        res_dict[ke] = {}
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
          _ul = result_or_simple_upper_limit( uavg )
          _ul_inclSys = result_or_simple_upper_limit( uavg_inclSys )
          if not type(_ul) == uncertainties.core.Variable:
            _uld = ufloat(_ul, _ul/10.) # define this quantity for easier printing
            _ulds =ufloat(_ul_inclSys, _ul_inclSys/10.)
          #print (f"{ke:>10}:  {uavg:S} Bq \t-> {uavg_inclSys:S} Bq; \t{is_bg_corrected=}")

          #if mass is None and pcs is None:
          if type(_ul) == uncertainties.core.Variable:
            print (f"{ke:>10}:  {uavg:S} Bq \t-> {uavg_inclSys:S} Bq")
            res_dict[ke]["activity"] = uavg
            res_dict[ke]["activity_inclSys"] = uavg_inclSys
          else:
            print (f"{ke:>10}:  <{_uld:S} Bq \t-> <{_ulds:S} Bq")
            res_dict[ke]["activity"] = _uld
            res_dict[ke]["activity_inclSys"] = _ulds
          #print (f"{ke:>10}:  {uavg:S} Bq \t-> {uavg_inclSys:S} Bq; \t{is_bg_corrected=}")
          #pass
          #elif not mass is None:
          if not mass is None:
            if type(_ul) == uncertainties.core.Variable:
            #print (f"{ke:>10}:  {uavg/mass:S} Bq/kg \t-> {uavg_inclSys/mass:S} Bq/kg; \t{is_bg_corrected=} {mass=} kg")
              print (f"{ke:>10}:  {uavg/mass:S} Bq/kg \t-> {uavg_inclSys/mass:S} Bq/kg")
              res_dict[ke]["activity_per_mass"] = uavg/mass
              res_dict[ke]["activity_per_mass_inclSys"] = uavg_inclSys/mass
            else:
              print (f"{ke:>10}:  <{_uld/mass:S} Bq/kg \t-> <{_ulds/mass:S} Bq/kg")
              res_dict[ke]["activity_per_mass"] = _uld/mass
              res_dict[ke]["activity_per_mass_inclSys"] = _ulds/mass

          #elif not pcs is None:
          if not pcs is None:
            if type(_ul) == uncertainties.core.Variable:
            #print (f"{ke:>10}:  {uavg/pcs:S} Bq/pc \t-> {uavg_inclSys/pcs:S} Bq/pc; \t{is_bg_corrected=} {pcs=} pieces")
              print (f"{ke:>10}:  {uavg/pcs:S} Bq/pc \t-> {uavg_inclSys/pcs:S} Bq/pc")
              res_dict[ke]["activity_per_pcs"] = uavg/pc
              res_dict[ke]["activity_per_pcs_inclSys"] = uavg_inclSys/pc
            else:
              print (f"{ke:>10}:  <{_uld/pcs:S} Bq/pc \t-> <{_ulds/pcs:S} Bq/pc")
              res_dict[ke]["activity_per_pcs"] = _uld/pc
              res_dict[ke]["activity_per_pcs_inclSys"] = _ulds/pc

          #else:
          #  print ("Something went wrong printing the results of {ke:>10}. {uavg:S} Bq; {mass=} {pcs=}")
#          if not type(_ul) == uncertainties.core.Variable:
#            print (f"{_ul=}  {type(_ul)=}")
#            print (f"\t\t -> Upper Limit: <{_uld:S} Bq  ; incl Sys: <{_ulds:S} Bq")
#            _uld = ufloat(_ul, _ul/10.) # define this quantity for easier printing
#            _ulds =ufloat(_ul_inclSys, _ul_inclSys/10.)

        except Exception as e:
            print (f"{ke:>10}: {e}")
    
    return res_dict
