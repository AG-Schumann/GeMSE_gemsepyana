import os.path

import uproot
import numpy as np
import matplotlib.pyplot as plt

class GeMSEData_uproot():
    ''' 
    This class is depreceated but still in use for the initial annotation of files
    - Sebastian: Sep 2024
    '''

    def __init__(self, fn, verbosity=False):
        self.fn = fn  # name of root file
        self.x = None # bin centers of ADC counts
        self.y = None # entries per bin
        self.edges = None # edges of histogram
        self.sample_name = None # verbous sample name
        self.t_real = None
        self.t_live = None
        self.verbose=verbosity
        self.iso_dict = self.read_parameters_files(
            absolute_path_to_parameters_dictionary=os.path.join(os.path.dirname(__file__), 'default_isotope_parameters')
            )
        self.manual_dict = None
        self.init_manual_dict()
        self.result_dict=None
        self._xlim = None
        self.bat_params = None
        self._respars = None
        self.y_sim = None
        self.fit_dict = None

    def read_bat_params(self, fn):
        ### fn is typically the "parameters_activity_calculation.txt" file used by "GeMSE_analysis"
        #iso_paras_path = "/home/sebastian/Computing/GeMSE/GeMSE_analysis/GeMSE_analysis/data/example/isotope_parameters"
        #from os import listdir
        from os.path import exists #, isfile, join
        if not exists(fn):
            print ("bat_params file does not exist. fn= '", fn, "'")
            return 0
        _d = {}
        _k = None
        with open(fn, 'r') as f:
            for line in f:
                if line.startswith('# '):
                    _k = line.strip('# ').strip()
                    _d[_k] = []
                else:
                    _d[_k].append(line.strip())
        self.bat_params = _d 

   # def get_bincenters(self, edges):
   #     _bc = [ (edges[i]+edges[i+1])/2. for i in range(len(edges)-1) ]
   #     return np.array( _bc )

    def get_bincenters(self):
        _bc = [ (self.edges[i]+self.edges[i+1])/2. for i in range(len(self.edges)-1) ]
        return np.array( _bc )

    def load_spectrum(self, thrl=10, thrh=3100, preview=False):

        if not os.path.isfile(self.fn):
            print (self.fn, ' does not exist.')
            raise FileNotFoundError('File does not exist.')

        with uproot.open(self.fn) as root_spectrum:
            hist = root_spectrum["hist"]
            h_edges = hist.axis().edges()
            h_cont = np.array(list(hist.values()))

        self.edges = h_edges
        #bc = self.get_bincenters(edges = h_edges)
        bc = self.get_bincenters()
        xlim = (bc>thrl) & (bc<thrh)

        self.x = bc[xlim]
        self.y = h_cont[xlim]

        if self.verbose:  
            print (f'len(xlim)={len(xlim)}')
            print (f'len(xb)={len(self.x)}')
            print (f'len(wy)={len(self.y)}')

        #f.Close()

        if preview:
            self.plot_spectrum(res_dict)

        return None


    def plot_spectrum(self, fig=None, ax=None, fn=None, xlim=None):

        if fig is None or ax is None:
            fig, ax = plt.subplots(figsize=(12,8))

        if xlim is not None:
            self._xlim = xlim

        ax.plot(self.x , self.y , '-', c='b', lw=1, label=self.sample_name)
        ax.set_xlabel('energy / keV')
        ax.set_ylabel('counts per bin / 1')
        ax.set_yscale('log')

        fig.legend(fontsize=12)
        fig.tight_layout()
        if not xlim is None:
            ax.set_xlim(*xlim)
        #fig.show()

        if fn:
            fig.savefig(fn)
        else:
            fig.show()

        return None
    

    def read_parameters_files(self, absolute_path_to_parameters_dictionary):
        #iso_paras_path = "/home/sebastian/Computing/GeMSE/GeMSE_analysis/GeMSE_analysis/data/example/isotope_parameters"
        from os import listdir
        from os.path import isfile, join
        #mport numpy as np
        _iso_dict = {}
        for fn in listdir(absolute_path_to_parameters_dictionary):
            iso_name = fn.strip('parameters_').strip('.txt')
            # print ('*** ' + iso_name + " ***")
            _d = {}
            _k = None
            with open(join(absolute_path_to_parameters_dictionary,fn), 'r') as f:
                for line in f:
                    if line.startswith('# '):
                        _k = line.strip('# ').strip()
                        _d[_k] = []
                    else:
                        _d[_k].append(float(line))
            _iso_dict[iso_name] = _d
        return _iso_dict 
    
    def init_manual_dict(self):
        self.manual_dict = {'Pb214': [(241.997, 7.43),],
               'Bi214': [(609.312, 46.1),
                         (665.453, 1.46),
                         (768.356, 4.94),
                         (934.061, 3.03),
                         (1120.287, 15.1),
                         (1238.110, 5.79), 
                         (1377.669, 4.00), 
                         (1407.98, 2.15),
                         (1509.228, 2.11),
                         (1729.595, 2.92),
                         (1764.494, 15.4),
                         (1847.420, 2.11),
                         (2118.55, 1.14),
                         (2203.85, 2.22),
                         (2204.21, 5.08),
                        ],
               'e+e-' : [(511.000, 1.0),],
               'Ra226' : [(186.211, 3.59),],
               'Pb212' : [(238.632, 43.3), 
                          (300.087,3.28),],
               'Bi212' : [(727.330, 6.58),
                         (785.37, 1.102),
                         (1620.50, 1.49),],
               'Al26'  : [(1808.72, 1.0),],
              }
        
    def iso_dict2simple_dict(self, iso_dict, default_BR=-1):
        _d = {}
        for _k in iso_dict.keys():
            _d[_k] = []
            for i in range(int(iso_dict[_k]['Number of peaks'][0])):
                _d[_k].append((iso_dict[_k]['Peak Energies (keV)'][i], default_BR ))
        return _d
            
    def draw_lines(self, 
                   add_labels=True, 
                   xrange=None, 
                   ke='Peak Energies (keV)', 
                   klf='Lower Fit Range (keV)', 
                   kuf='Upper Fit Range (keV)',
                   col='k',
                   do_count=True,
                   ):
        cax = plt.gca()
        _pos = 0.85
        _old_pos = 0.82
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        for key in self.iso_dict:
            for idx in range(len(self.iso_dict[key][ke])):
                enrg = self.iso_dict[key][ke][idx]
                lfr  = self.iso_dict[key][klf][idx]            
                ufr  = self.iso_dict[key][kuf][idx]
                if xrange is None or (enrg > xrange[0] and enrg< xrange[1]):
                    #print (f'{key}: {enrg}')
                    cax.axvline(x=enrg, color=col, ls='--', lw=1)
                    cax.axvline(x=lfr, color=col, ls='--', lw=0.5)
                    cax.axvline(x=ufr, color=col, ls='--', lw=0.5)

                    trans = cax.get_xaxis_transform()
                    if add_labels:
                        _t_pos = _old_pos
                        _old_pos = _pos
                        _pos = _t_pos
                        
                        if not do_count:
                            annotation_str = key
                        else:
                            _cts, _Xcts = self._integrate_range( xrange=(lfr,ufr) )
                            annotation_str = f'{key} ({_cts} cts)'
                            cax.fill_between(self.x[_Xcts], self.y[_Xcts],facecolor='blue', alpha=0.5)
                        
                        cax.text(enrg, _pos, f'{annotation_str}', 
                                 horizontalalignment='center', 
                                 verticalalignment='center', 
                                 transform=trans,
                                 bbox=dict(facecolor='white', edgecolor='none'))
                        if self.result_dict:
                            #rate = result_dict[key]['rate background corrected'][idx]
                            #rate_unc = result_dict[key]['rate background corrected uncertainty'][idx]
                            _cfbg = self.result_dict[key]['compton background range'][idx]
                            #print (f'{_cfbg}')
                            cax.axvline(x=_cfbg[0], color=col, ls='-.', lw=0.3)
                            cax.axvline(x=_cfbg[1], color=col, ls='-.', lw=0.3)

                            cax.plot((_cfbg[2],_cfbg[4]), (_cfbg[3],_cfbg[5]), ls='--', color='m' )
                            _cc = self.result_dict[key]['corrected counts'][idx]
                            cax.text(enrg, _pos-0.15, f'{_cc[0]:.1f} +- {_cc[1]:.1f}cts', 
                                     horizontalalignment='center', 
                                     verticalalignment='center', 
                                     transform=trans,
                                     fontsize=8,
                                     bbox=dict(facecolor='white', edgecolor='none'))

                            #cax.text(enrg, _pos-0.05, rate2str(rate, rate_unc), 
                            #         horizontalalignment='center', 
                            #         verticalalignment='center', 
                            #         transform=trans,
                            #         fontsize=8,
                            #  
                            
    def draw_special_lines(self, sdict=None, add_labels=True, xrange=None, col='b', ax=None):
        if ax is None:
            cax = plt.gca()
        else:
            cax = ax
        _pos = 0.77 -0.05
        _old_pos = 0.74 - 0.05
        
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        
        if sdict is None:
            sdict = self.manual_dict
        for iso, infos in sdict.items():
            for _inf in infos:
                #print (_inf)
                try:
                    enrg, BR = _inf
                except Exception as e:
                    print (f'Exception: {e}\n_inf={_inf} (iso={iso})')
                if xrange is None or (enrg > xrange[0] and enrg< xrange[1]):
                    #print (f'{key}: {enrg}')
                    cax.axvline(x=enrg, color=col, ls='--', lw=1)
                    #cax.axvline(x=lfr, color=col, ls='--', lw=0.5)
                    #cax.axvline(x=ufr, color=col, ls='--', lw=0.5)

                    trans = cax.get_xaxis_transform()
                    if add_labels:
                        _t_pos = _old_pos
                        _old_pos = _pos
                        _pos = _t_pos
                        cax.text(enrg, _pos, f'{iso} (BR {BR})', 
                                 horizontalalignment='left', 
                                 verticalalignment='center', 
                                 transform=trans, 
                                 color=col, 
                                 bbox=dict(facecolor='white', edgecolor='none'))


    def _get_nsig_intervals(self, mean, sigma, nlevels=3, x=None, y=None):
        if x is None:
            x=self.x
        if y is None:
            y=self.y
        ### let's count entries in +-levels*sigma around mean:
        intervals = np.arange(mean-nlevels*sigma, mean+(nlevels+1)*sigma, sigma)
        ## https://en.wikipedia.org/wiki/File:Standard_deviation_diagram.svg
        return intervals                    
                        
                        
    def _integrate_range(self, xrange, x=None, y=None):
        if x is None:
            x=self.x
        if y is None:
            y=self.y
        ## TODO: do we want to interpolate? Or what to do in the (standard) case where the range does not coincide with bins?
        _Xr = ((x>xrange[0]) & (x<xrange[1])) ## TODO: Fix (see above)!
        return y[_Xr].sum(), _Xr
    
#    def annotate_counts(self, todo_dict):
#        cax = plt.gca()
#        for iname in todo_dict:
#            mean, sigma, nlevels = todo_dict[iname]
#            _intervals = self._get_nsig_intervals(mean=mean, sigma=sigma, nlevels=nlevels)
#            _nsig=-nlevels
#            _counts = [self._integrate_range(xrange=_intervals[i:i+2])[0] for i in range(2*nlevels) ]
#            for i in range(2*nlevels):
#                print (f'{_nsig} sigma: {_intervals[i:i+2]} -> {_counts[i]} cts')
#                _nsig += 1 

                        
    def create_pdf_summary(self, outfn=None):
        xrange_full = np.arange(0,3600,150)
        
        from matplotlib.backends.backend_pdf import PdfPages

        if outfn is None:
            outfn = f'{self.sample_name}_summary_raw.pdf'
            
        with PdfPages(outfn) as pdf:
            for ir in range(len(xrange_full)-1):
                plt.figure(figsize=(12, 8))

                xrange = xrange_full[ir:ir+2]
                title_str = f'{xrange[0]} keV - {xrange[1]} keV'
                print (title_str)
                plt.title(title_str)


            
                plt.plot(self.x, self.y, lw=2, ls='-', c='k', label=f'{self.sample_name}')
                #plt.errorbar(self.x, self.y, yerr=np.sqrt(self.y), lw=2, ls='-', c='k', label=f'{self.sample_name}')

                self.draw_lines(add_labels=True, xrange=xrange)
                #
                self.draw_special_lines(add_labels=True, xrange=xrange, col='m')


                if not xrange is None:
                    plt.gca().set_xlim(*xrange)

                plt.gca().set_yscale('log')
                plt.gca().set_xlabel('Energy / keV')
                plt.gca().set_ylabel('cts/bin')

                plt.legend()
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()


    def resolution_function(self, x):
        ## uproot seems not to be able to extract a TF1 from a TCanvas (as it is produced by GeMSE_analysis)
        ## let's do it manually for now...
#         import os
#         if self.bat_params is None:
#            #self.read_bat_params()
#            print ("You have to load a 'bat_params' file. Exiting. ")
#            return 0
#        _rf = self.bat_params["resolution file name"][0]
#
#        if not os.path.isfile(_rf):
#            print (_rf, ' does not exist.')
#            raise FileNotFoundError('File does not exist.')
#        
#        with uproot.open(_rf) as root_resolution:
#            hist = root_resolution["hist"]
#            h_edges = hist.axis().edges()
#            h_cont = np.array(list(hist.values()))

        ##########  
        # resolution function is: sqrt(([0] +([1]*x))+([2]*(x*x)))
        #return np.sqrt( (_respars[0] +(_respars[1]*x))+(_respars[2]*(x*x)) )
        ######
        # Please define when calling!
        ######
        pass

    def fit_gauss_p0(self, xrange, p0=None ):
        #  xrange = (lfr-_Dlu, ufr+_Dlu)
        #  p0 = (_cts_corr, enrg, _Drf, _av_bg)
        from scipy.optimize import curve_fit
        ## Attention! number of counts cts = A*binwidth
        # _func = lambda x, p0, p1, p2, p3: p0/np.sqrt(2*np.pi*p2**2) * np.exp(-(x-p1)**2/(2*p2**2) ) + p3
        _func_constr = lambda x, p_0, p_3: p_0/np.sqrt(2*np.pi*p0[2]**2) * np.exp(-(x-p0[1])**2/(2*p0[2]**2) ) + p_3
        _Xr = ((self.x>xrange[0]) & (self.x<xrange[1]))
        _popt, _pcov = curve_fit(_func_constr, self.x[_Xr], self.y[_Xr], p0=(p0[0],p0[3]), bounds=((0,0),(np.inf,np.inf))  )
        return _popt, _pcov


    def fit_spectrum(self, ke='Peak Energies (keV)', klf='Lower Fit Range (keV)', kuf='Upper Fit Range (keV)',):
        for key in self.iso_dict:
            for idx in range(len(self.iso_dict[key][ke])):
                enrg = self.iso_dict[key][ke][idx]
                lfr  = self.iso_dict[key][klf][idx]            
                ufr  = self.iso_dict[key][kuf][idx]
                # Let's compare the 'standard' range with what we get using the resolution function:
                # The resolution function (TODO) currently has to be provided from the (external) calling script that manually reads the resolution_function.root
                _Drf = self.resolution_function(x=enrg)
                _cts, _Xcts = self._integrate_range( xrange=(lfr,ufr) )
                ## rough estimate of binwidth:
                _bw = (self.x[_Xcts][1]-self.x[_Xcts][0])
                _Dlu = ufr-lfr
                # flat bg_left:
                # integrate same interval width to the left
                _cts_left, __Xctsl = self._integrate_range( xrange=(lfr-_Dlu, lfr) )
                # flag bg_right:
                # integrate same interval width to the right
                _cts_right, __Xctsr = self._integrate_range( xrange=(ufr, ufr+_Dlu) )
                # average flat bg:
                _av_bg = (_cts_left+_cts_right)/2.
                _av_bg_rate = _av_bg/_Dlu * _bw 
                _cts_corr = _cts - _av_bg
                
               
                # Fit peak with gauss + flat bg:
                res_fit_gauss_p0, cov_fit_gauss_p0 = self.fit_gauss_p0( xrange=(lfr-_Dlu, ufr+_Dlu), p0=(np.maximum(_cts_corr, 0), enrg, _Drf, _av_bg) )
                _cts_fit = res_fit_gauss_p0[0]/_bw
                _flat_bg_fit = res_fit_gauss_p0[1]/_bw

                print (f'flat bg rate [{lfr-_Dlu:.1f}-{ufr+_Dlu:.1f}]keV: {_cts:.1f}cts ({_av_bg_rate:.1f}cts/keV)')
                print (f'              fit: {_cts_fit:.1f}cts ({_flat_bg_fit:.1f}cts/keV)')



    def draw_gauss_p0(self, A, mu, sig, binwidth=None, nsig=5, p0=None, color='r'): 
        cax = plt.gca()
        _x = np.linspace(mu-nsig*sig, mu+nsig*sig, num=100)
        _y2 = A/np.sqrt(2*np.pi*sig**2) * np.exp(-(_x-mu)**2/(2*sig**2) )
        if not p0 is None:
          _y2 += p0
        ## account for binning:
        if not binwidth is None:
          # binwidth=1
          _y2 = _y2*binwidth
        cax.plot(_x, _y2, color=color, ls='--', lw=2)



    def draw_lines_bat(self, 
                   add_labels=True, 
                   xrange=None, 
                   ke='Peak Energies (keV)', 
                   klf='Lower Fit Range (keV)', 
                   kuf='Upper Fit Range (keV)',
                   col='k',
                   do_count=True,
                   ):
        cax = plt.gca()
        _pos = 0.85
        _old_pos = 0.82
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        for key in self.iso_dict:
            for idx in range(len(self.iso_dict[key][ke])):
                enrg = self.iso_dict[key][ke][idx]
                if xrange is None or (enrg > xrange[0] and enrg< xrange[1]):
                    lfr  = self.iso_dict[key][klf][idx]            
                    ufr  = self.iso_dict[key][kuf][idx]
                    # Let's compare the 'standard' range with what we get using the resolution function:
                    # The resolution function (TODO) currently has to be provided from the (external) calling script that manually reads the resolution_function.root
                    _Drf = self.resolution_function(x=enrg)
                    # lfr_rf = enrg-_Drf
                    # ufr_rf = enrg+_Drf
                    # print (f'enrg={enrg:.1f}: lfr(lfr_rf)={lfr:.1f}({lfr_rf:.1f}),  ufr(ufr_rf)={ufr:.1f}({ufr_rf:.1f})')

                    #print (f'{key}: {enrg}')
                    cax.axvline(x=enrg, color=col, ls='--', lw=1)
                    cax.axvline(x=lfr, color=col, ls='--', lw=0.5)
                    cax.axvline(x=ufr, color=col, ls='--', lw=0.5)
                    #cax.axvline(x=lfr_rf, color=col, ls='--', lw=1.5)
                    #cax.axvline(x=ufr_rf, color=col, ls='--', lw=1.5)

                    trans = cax.get_xaxis_transform()
                    if add_labels:
                        _t_pos = _old_pos
                        _old_pos = _pos
                        _pos = _t_pos
                        
                        if not do_count:
                            annotation_str = key
                        else:
                            _cts, _Xcts = self._integrate_range( xrange=(lfr,ufr) )
                            ## rough estimate of binwidth:
                            _bw = (self.x[_Xcts][1]-self.x[_Xcts][0])
                            _Dlu = ufr-lfr
                            # flat bg_left:
                            # integrate same interval width to the left
                            _cts_left, __Xctsl = self._integrate_range( xrange=(lfr-_Dlu, lfr) )
                            # flag bg_right:
                            # integrate same interval width to the right
                            _cts_right, __Xctsr = self._integrate_range( xrange=(ufr, ufr+_Dlu) )
                            # average flat bg:
                            _av_bg = (_cts_left+_cts_right)/2.
                            _av_bg_rate = _av_bg/_Dlu * _bw 
                            _cts_corr = _cts - _av_bg
                            
                            cax.fill_between(self.x[_Xcts], self.y[_Xcts],facecolor='blue', alpha=0.5)
                           
                            # Fit peak with gauss + flat bg:
                            res_fit_gauss_p0, cov_fit_gauss_p0 = self.fit_gauss_p0( xrange=(lfr-_Dlu, ufr+_Dlu), p0=(np.maximum(_cts_corr, 0), enrg, _Drf, _av_bg) )
                            _cts_fit = res_fit_gauss_p0[0]/_bw
                            _flat_bg_fit = res_fit_gauss_p0[1]/_bw

                            print (f'flat bg rate [{lfr-_Dlu:.1f}-{ufr+_Dlu:.1f}]keV: {_cts:.1f}cts ({_av_bg_rate:.1f}cts/keV)')
                            print (f'              fit: {_cts_fit:.1f}cts ({_flat_bg_fit:.1f}cts/keV)')
                            cax.hlines(y=_av_bg_rate, xmin=lfr-_Dlu, xmax=ufr+_Dlu, linestyles='-', linewidths=1, colors='g')
                            #cax.axhline(y=_av_bg_rate, xmin=lfr-_Dlu, xmax=ufr+_Dlu, ls='-', lw=1, color='g')
                            
                            annotation_str = f'{key} ({_cts:.0f}(bg:{_av_bg:.1f}) fit:{_cts_fit:.1f}(bg:{_flat_bg_fit:.1f}) cts)'
                            
                            # Draw gaussians:
                            # self.draw_gauss_p0(A=_cts_corr, mu=enrg, sig=_Drf, binwidth=_bw, p0=_av_bg_rate, color='r' )
                            #self.draw_gauss_p0(A=res_fit_gauss_p0[0]/_bw, mu=enrg, sig=_Drf, binwidth=_bw, p0=res_fit_gauss_p0[1]*_bw, color='m' )
                            # self.draw_gauss_p0(A=res_fit_gauss_p0[0], mu=enrg, sig=_Drf, binwidth=_bw, p0=res_fit_gauss_p0[1], color='m' )
                            self.draw_gauss_p0(A=_cts_fit, mu=enrg, sig=_Drf, binwidth=_bw, p0=_flat_bg_fit, color='m' )
                        
                        cax.text(enrg, _pos, f'{annotation_str}', 
                                 horizontalalignment='center', 
                                 verticalalignment='center', 
                                 transform=trans,
                                 bbox=dict(facecolor='white', edgecolor='none'))
                        if False:
                        # TODO: check the following:
                        #if self.result_dict:
                            #rate = result_dict[key]['rate background corrected'][idx]
                            #rate_unc = result_dict[key]['rate background corrected uncertainty'][idx]
                            _cfbg = self.result_dict[key]['compton background range'][idx]
                            #print (f'{_cfbg}')
                            cax.axvline(x=_cfbg[0], color=col, ls='-.', lw=0.3)
                            cax.axvline(x=_cfbg[1], color=col, ls='-.', lw=0.3)

                            cax.plot((_cfbg[2],_cfbg[4]), (_cfbg[3],_cfbg[5]), ls='--', color='m' )
                            _cc = self.result_dict[key]['corrected counts'][idx]
                            cax.text(enrg, _pos-0.15, f'{_cc[0]:.1f} +- {_cc[1]:.1f}cts', 
                                     horizontalalignment='center', 
                                     verticalalignment='center', 
                                     transform=trans,
                                     fontsize=8,
                                     bbox=dict(facecolor='white', edgecolor='none'))

                            #cax.text(enrg, _pos-0.05, rate2str(rate, rate_unc), 
                            #         horizontalalignment='center', 
                            #         verticalalignment='center', 
                            #         transform=trans,
                            #         fontsize=8,
                            #  
                            
            
     

    def simulate_spectrum(self, cts, enrg, res, flat_bg, xrange=(0,3600), thrl=10, thrh=3100):
        from scipy.stats import norm
        from numpy.random import uniform
        evts = np.array( [] )
        for _idx in range(len(cts)):
          c = cts[_idx]
          e = enrg[_idx]
          r = res[_idx]
          fb = flat_bg[_idx]
          print (f'N={c} E={e}keV res={r}keV flat_bg={fb}')
          
          # simulate gaussian peaks
          _gsim = norm.rvs(loc=e, scale=r, size=int(c))
          evts = np.concatenate( (evts,_gsim), axis=None )
          
          # simulate 'flat compton background' up to peak energy - a bit
          _true_rate = fb
          nce=15
          n_bg = _true_rate * (e-nce*r-thrl)
          _flatbgsim = uniform(thrl,e-nce*r, int(n_bg) )
          evts = np.concatenate( (evts, _flatbgsim), axis=None )


        hsim, hsim_edges = np.histogram( evts, bins=self.edges )

        bc = self.get_bincenters()
        xlim = (bc>thrl) & (bc<thrh)

        self.x = bc[xlim]
        self.y_sim = hsim[xlim]


    def create_pdf_summary_bat(self, outfn=None, stepsize=150, Emin=0, Emax=3600, logy=True, errorbar=False):
        xrange_full = np.arange(Emin,Emax,stepsize)
        if len(xrange_full) < 3:
            _xrange_full = np.array( [xrange_full[0], xrange_full[0]+stepsize, xrange_full[0]+2*stepsize] )
            xrange_full = _xrange_full
        
        from matplotlib.backends.backend_pdf import PdfPages

        if outfn is None:
            outfn = f'{self.sample_name}_summary_bat.pdf'
            
        with PdfPages(outfn) as pdf:
            for ir in range(len(xrange_full)-1):
                plt.figure(figsize=(12, 8))

                xrange = xrange_full[ir:ir+2]
                title_str = f'{xrange[0]} keV - {xrange[1]} keV'
                print (title_str)
                plt.title(title_str)

                if errorbar:
                  # schoener, aber viel langsamer:
                  plt.errorbar(self.x, self.y, yerr=np.sqrt(self.y), lw=2, ls='-', c='k', label=f'{self.sample_name}')
                else:
                  plt.plot(self.x, self.y, lw=2, ls='-', c='k', label=f'{self.sample_name}')

                ## TODO 
                # self.draw_lines(add_labels=True, xrange=xrange)
                self.draw_lines_bat(add_labels=True, xrange=xrange)
                #
                ## 
                # TODO self.draw_special_lines(add_labels=True, xrange=xrange, col='m')


                if not xrange is None:
                    plt.gca().set_xlim(*xrange)
                plt.gca().set_ylim(0.1, None)
                
                if logy:
                    plt.gca().set_yscale('log')
                plt.gca().set_xlabel('Energy / keV')
                plt.gca().set_ylabel('cts/bin')

                plt.legend()
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
