import os.path

import uproot
import numpy as np
import matplotlib.pyplot as plt
from os import path

class GeMSEData_bat():
    ''' 
    This class is intendend to plot GeMSE
    histograms USING UPROOT and overlay BAT results.
    - Sebastian Sep 2024: package based on intial gemsedata_bat file --
    '''

    def __init__(self, fn=None, verbosity=False):
        self.fn = fn  # name of root file; can be set from the bat_parameters file
        self.x = None # bin centers of ADC counts
        self.y = None # entries per bin
        self.edges = None # edges of histogram
        self.binwidth = None
        self.sample_name = None # verbous sample name
        self.t_real = None
        self.t_live = None
        self.verbose=verbosity
        self.iso_dict = self.read_parameters_files(
            #absolute_path_to_parameters_dictionary="default_isotope_parameters"
            absolute_path_to_parameters_dictionary=path.join(path.dirname(__file__), 'default_isotope_parameters')
            )  # default parameters will be overwritten, if bat parameter file is specified.
        #self.iso_dict = None # Is being set once bat parameter file is read
        self.manual_dict = None
        self.init_manual_dict()
        self.init_decay_chains()
        self.result_dict=None
        self._xlim = None
        self.bat_params = None
        self._respars = None
        self.y_sim = None
        self.fit_dict = None
        # Todo: make this more flexible:
        self.sampledict_txt="bkgdict.txt" 
        self.bkgdict_txt="sampledict.txt"
        self.bkgtimes_txt="bkgtimes.txt"
        self.sampletimes_txt="sampletimes.txt"
        #self.eff_rootfile="/home/sebastian/Computing/GeMSE/my_mc_results/arrowhead/merged_mc_results.root.root"
        self.eff_rootfile=None # "/home/sebastian/Computing/GeMSE/my_mc_results/arrowhead/merged_mc_results.root.root"
        self.eff_dict = None # self.get_efficiencies()
        ## BAT results
        self.bat_info, self.bat_vals, self.bat_uls = None, None, None
        #self.load_bat_results(fn=self.bat_results_path)
        self.resolution_function = None
        self.bat_summary = None
        self.x_orig, self.y_orig, self.bw_orig, self.ynorm_orig = None, None, None, None
        self._path_to_container_image = "/home/sebastian/Computing/GeMSE/singularity_containers/gemse_env_sl_wand.simg"

    def load_bat_params(self, fn):
        ### fn is typically the "parameters_activity_calculation.txt" file used by "GeMSE_analysis"
        #iso_paras_path = "/home/sebastian/Computing/GeMSE/GeMSE_analysis/GeMSE_analysis/data/example/isotope_parameters"
        #from os import listdir
        print ("Called 'load_bat_params(fn=", fn, "')")
        from os.path import exists #, isfile, join
        if not exists(fn):
            print ("bat_params file does not exist. fn= '", fn, "'")
            return 0
        _d = {}
        _k = None
        with open(fn, 'r') as f: # open files (doesnt need to be closed again because of the "with open()" function)
            for line in f: # going through all lines in the file
                #print (f"line={line}")
                if line.startswith('# '):
                    _k = line.strip('# ').strip() # removing the "# " and unnecessary " " and start and end
                    _d[_k] = []
                else:
                    _d[_k].append(line.strip())
        self.bat_params = _d
        #print (f"bat_params: {self.bat_params}")
        self.fn = self.bat_params['sample spectrum file name'][0]
        self.eff_rootfile = self.bat_params['efficiency file name'][0]
        self.iso_dict = self.read_parameters_files(absolute_path_to_parameters_dictionary=self.bat_params['isotope parameters folder'][0])
        self.eff_dict = self.get_efficiencies()
        self.set_resolution_function(self.bat_params['resolution file name'][0])
        self.bat_summary=self.bat_params['results folder'][0] + '/' + self.bat_params['sample name'][0] + '_activities_summary.txt'
        self.sample_name = self.bat_params['sample name'][0]
        print ('BAT parameters loaded.')
        print ('  Bat_params:', self.bat_params)
        print ('  I will annotate the file: ', self.fn )

    def load_bat_results(self):
        from analysis_classes.read_bat_summary import read_bat_summary as rbs
        self.bat_info, self.bat_vals, self.bat_uls = rbs(fn=self.bat_summary)
        #self.bat_params=self.load_bat_params(self.bat_info)
        print ('Bat results loaded.')
        print (self.bat_info, self.bat_vals, self.bat_uls)

    def get_bincenters(self): # calculating bin center from 1 to i
        _bc = [ (self.edges[i]+self.edges[i+1])/2. for i in range(len(self.edges)-1) ]
        return np.array( _bc ) # saving in an array

    def get_binwidth(self): # calculating bin widths
        _bw = [ (self.edges[i+1]-self.edges[i]) for i in range(len(self.edges)-1) ]
        return np.array( _bw ) # saving in an array


    def rebin(self, nbins=10):
      ### Seba 12 Sep 2024: need to check this function. Summing y-bins in a fixed x-range seems to not result in same/similar counts. But this should be the case!
      _e1 = 1 # added 13 Sep 2024
      _e2 = 0 # added 13 Sep 2024
      _e3 = -1 # added 13 Sep 2024
      if nbins > 1:
        xn, yn, bwn, ynorm = [], [], [], []
        if self.x_orig is not None:
          self.x = self.x_orig
          self.y = self.y_orig
          self.binwidth = self.bw_orig
          self.y_per_keV_per_day = self.ynorm_orig

        for i in range(0, len(self.x)-nbins+_e1, nbins): # check! we seem to miss some bins, no?!
            _bw = np.sum ( self.binwidth[i:i+nbins+_e2] )
            _x = ( (self.x[i]-self.binwidth[i])+(self.x[i+nbins+_e3]+self.binwidth[i+nbins+_e3]) )/2.
            ## can the following be correct? Why should the count number in a bin after rebinning care for the with of old/new bins?!
            ## _y = np.sum( self.y[i:i+nbins-1]*self.binwidth[i:i+nbins-1] )/ _bw ## uncommented on Sep 12, 2024 by SL
            _y = np.sum ( self.y[i:i+nbins+_e2] )
            _ynorm = np.sum( self.y_per_keV_per_day[i:i+nbins+_e2]*self.binwidth[i:i+nbins+_e2] )/ _bw
            xn.append(_x)
            yn.append(_y)
            bwn.append(_bw)
            ynorm.append(_ynorm)
            
        self.x = np.array(xn)
        self.y = np.array(yn)
        self.binwidth = np.array(bwn)
        self.y_per_keV_per_day = np.array(ynorm)
      else:
        self.x = self.x_orig
        self.y = self.y_orig
        self.binwidth = self.bw_orig
        self.y_per_keV_per_day = self.ynorm_orig

    def norm_cts_kev_day(self):
      if self.t_live is None:
          _ltdays = 1
          print ("Attention! Lifetime not provided. I won't normalise by time!")
      else:
          _ltdays = self.t_live/(3600*24)
      _nck = np.array( [ self.y[i]/self.binwidth[i]/_ltdays for i in range(len(self.y)) ] )
      #print (f"type(_nck) = {type(_nck)}")
      return _nck

    def load_spectrum(self, thrl=10, thrh=3100, preview=False):

        if not os.path.isfile(self.fn): # Error if no file
            print (self.fn, ' does not exist.')
            raise FileNotFoundError('File does not exist.')

        print ("Will now load file: ", self.fn)

        with uproot.open(self.fn) as root_spectrum:
            hist = root_spectrum["hist"]
            h_edges = hist.axis().edges()
            h_cont = np.array(list(hist.values()))

        self.edges = h_edges
        #bc = self.get_bincenters(edges = h_edges)
        bc = self.get_bincenters()
        bw = self.get_binwidth()
        xlim = (bc>thrl) & (bc<thrh)

        self.x = bc[xlim]
        self.y = h_cont[xlim]
        self.binwidth = bw[xlim]
        #print ("should now call self.norm_cts_kev_day()")
        self.y_per_keV_per_day = self.norm_cts_kev_day()

        ## keep original histogram in case of rebinning
        self.x_orig = self.x
        self.y_orig = self.y
        self.bw_orig = self.binwidth
        self.ynorm_orig = self.y_per_keV_per_day

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
        #import numpy as np
        _iso_dict = {}
        for fn in listdir(absolute_path_to_parameters_dictionary): # going through all isotope files
            iso_name = fn.strip('parameters_').strip('.txt') # extracting isotope name
            #print ('*** ' + iso_name + " ***")
            _d = {}
            _k = None
            if len(iso_name) > 6:
              print (f" ... read_parameters_files ... something wrong with this file: {fn}! Is it of type 'parameters_XYZ.txt'?")
              print ("  ... will ignore this file and continue with the next.")
              continue
            with open(join(absolute_path_to_parameters_dictionary,fn), 'r') as f: # open the files (absolute_path_to_parameters_dictionary is path, fn is file)
                for line in f:
                    if line.startswith('# '):
                        _k = line.strip('# ').strip()
                        _d[_k] = []
                    else:
                        _d[_k].append(float(line))
            _iso_dict[iso_name] = _d
            #def get_energies(item):
            #    energies = item[1]["Peak Energies (keV)"]
            #    return energies[0]
        #return sorted(_iso_dict.items(), key=get_energies)
        return _iso_dict

    def init_decay_chains(self):
        self.decay_chains = {
            'Th232_full': ['232Th','228Ra','228Ac','228Th','224Ra','220Rn','216Po','212Pb','212Bi','208Tl','212Po','208Tl','208Pb'],
            'U238_full': ['238U', '234Th', '234mPa', '234Pa', '234U', '230Th', '226Ra', '222Rn', '218Po', '218At', '218Rn', '214Pb', '214Bi', '214Po', '210Tl', '210Pb', '210Bi', '210Po', '206Hg', '206Tl', '206Pb',],
            'U235_full': ['235U', '231Th', '231Pa', '227Ac', '227Th', '223Fr', '223Ra', '219At', '219Rn', '215Bi', '215Po', '215At', '211Pb', '211Bi', '211Po', '207Tl', '207Pb', ],
            'U238_early': ['238U', '234Th', '234mPa', '234Pa', '234U', '230Th',],
            'U238_intermediate': [ '226Ra', '222Rn', '218Po', '218At', '218Rn', '214Pb', '214Bi', '214Po', '210Tl', ],
            'U238_late': [ '210Pb', '210Bi', '210Po', '206Hg', '206Tl', '206Pb',],
            'U235_early': ['235U', '231Th'],
            'U235_late': ['231Pa', '227Ac', '227Th', '223Fr', '223Ra', '219At', '219Rn', '215Bi', '215Po', '215At', '211Pb', '211Bi', '211Po', '207Tl', '207Pb'],
            }
    
    def init_manual_dict(self): # manually added isotopes with parameters
        self.manual_dict = {
               'Pb214': [( 351.932, 37.6), (295.224, 19.3), (241.997, 7.43), 
                         (53.2275, 1.2), (785.96, 1.07), (839.04, 0.587), 
                         (258.87, 0.524), (274.80, 0.474), (487.09, 0.422), 
                         (580.13, 0.352), (480.43, 0.320), (462.00, 0.221), 
                         (533.66, 0.186), 
                         ],
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
               '234mPa': [(1001.03, 0.837), (766.38, 0.294), (742.81, 0.080), (258.23, 0.0728), 
                          (786.27, 0.0485), (1737.73, 0.0211), (1831.5, 0.0172), (1193.77, 0.01347), 
                          (73.92, 0.013), (1510.5, 0.01287), (921.70, 0.01269), (739.95, 0.0117), 
                          (946.00, 0.0099), (1435.36, 0.00968), (1867.68, 0.00918), (1765.44, 0.00868), 
                          (1875.5, 0.00818), (1554.1, 0.00808),],
               'e+e-' : [(511.000, 100.0),],
               'Ra226' : [(186.211, 3.59), (262.27, 0.0050)],
               'Pb212' : [(238.632, 43.3), (300.087, 3.28), (115.183, 0.592), (415.2, 0.143), (176.68, 0.052)],
               'Bi212' : [(727.330, 6.58),
                         (785.37, 1.102),
                         (1620.50, 1.49),
                         (39.858, 1.091), (1078.62, 0.564), (893.408, 0.378), (452.83, 0.31), (288.07, 0.31), (1512.7, 0.29),
                         ],
               'Al26'  : [(1808.72, 1.0),],
               'Tl208' : [(510.77, 22.0),
                          (583.191, 83.2),
                          (860.564, 12.5),
                          (2614.533, 100.0), ## energy is high enough for pair production, if 1x 511 keV from b+ annihilition escapes, we have another (DE) peak at 2614.5-511 = 2103.5 keV
                          (2103.5, 0.0000001), ## 1x escape peak (DE) from 2.6 MeV line
                          (1592.5, 0.0000001), ## 2x escape peak (DE) from 2.6 MeV line
                          (277.351, 6.31), (763.13, 1.81), (252.61, 0.69), (1093.9, 0.40), (233.36, 0.307),
                         ],
               'Co57'  : [(122.0614, 85.60),
                          (136.4743, 10.68),
                         ],
               'Co58'  : [(810.775, 99.), 
                          (863.959, 0.683), 
                          (1674.730, 0.518),
                          ],
               'Co60'  : [(1332.501, 99.9856),
                          (1173.237, 99.9736),
                          ],
               'K40'   : [(1460.830, 10.55),],
               'Mn54'  : [(834.848, 99.976),],
               'Th234' : [(63.29, 4.8), (92.38, 2.81), (92.80, 2.77), (112.81, 0.277),],
               'Th228' : [(84.373, 1.22), (215.983, 0.254), (131.613, 0.1305), (166.410, 0.1036),],
               'U235' : [(185.712, 57.2), (143.764, 10.96), (163.358, 5.08), (205.309, 5.01), (109.16, 1.54), (202.111, 1.08), (194.94, 0.63), (182.52, 0.34),],
               'Ti44'  : [(78.337, 96), (67.875, 94.4),],
               'Ac228' : [(911.204, 25.8),
                          (968.971, 15.8),
                          (338.320, 11.27),
                          (964.766, 4.99),
                          (463.004, 4.40),
                          (794.947, 4.25),
                          (209.253, 3.89),
                          (270.245, 3.46),
                          (1588.19, 3.22),
                          (328.000, 2.95),
                          (129.065, 2.42),
                          (409.462, 1.92),
                          (835.710, 1.61),
                          (1630.627, 1.51),
                          (772.291, 1.49),
                          (99.509, 1.26),
                          (755.315, 1.00),
                          (840.377, 0.91),
                          (562.500, 0.87),
                          (1495.93, 0.86),
                          (1459.138, 0.83),
                          (904.19, 0.77),
                          (153.977, 0.722),
                          (726.863, 0.62),
                          (1580.53, 0.60),
                          (830.486, 0.540),
                          (1247.08, 0.50),
                          (782.142, 0.485),
                          (57.766, 0.47),
                          (1638.281, 0.47),
                          (1501.57, 0.46),
                          (508.959, 0.45),
                          (332.370, 0.40),
                          (340.98, 0.369),
                          (199.407, 0.315),
                          (1110.610, 0.285),
                          (958.61, 0.28),
                          ],
                }
        
    def iso_dict2simple_dict(self, iso_dict, default_BR=-1):
        _d = {}
        for _k in iso_dict.keys():
            _d[_k] = []
            for i in range(int(iso_dict[_k]['Number of peaks'][0])):
                _d[_k].append((iso_dict[_k]['Peak Energies (keV)'][i], default_BR ))
        return _d
            

    def get_efficiencies(self, ke='Peak Energies (keV)'):
        print ("Calling 'get_efficiencies' ...")
        # Reading efficencies from ROOT files via C++ file "get_times_from_rootfile.cxx" and saving it in a txt
        import subprocess
        _tmpfile = "efficiencies.txt"
        _path_to_compiled_app = "/home/sebastian/Computing/GeMSE/GeMSE_root_scripts/GeMSE_ROOT_scripts/get_efficiencies_from_rootfile"
        with open(_tmpfile, "w") as f:
            #p1 = subprocess.run([_path_to_compiled_app, self.eff_rootfile], stdout=f, text=True)
            p1 = subprocess.run(["singularity","exec",self._path_to_container_image,_path_to_compiled_app, self.eff_rootfile], stdout=f, text=True)
        print ("Efficiencies loaded.")
        eff_dict = {}
        eff_dict_err = {}
        eff_noBR_dict = {}
        energy = None
        k = None
        l = None
        le = None
        enbr = None
        with open(_tmpfile, "rt") as f:                         # efficiencies.txt looks like
            for line in f:                                      ############################
                if line.startswith('#energy:'):                 # "#energy:"energy         #
                    k = line.strip("#energy:").strip()          # "#efficiency:"efficiency #
                    #print(k)                                   # "#eff_err:"eff_err       #
                    energy = float(k)                           # "#eff_BR:"eff_BR         #
                elif line.startswith('#eff_BR:'):               # "#energy:"energy         #
                    l = line.strip("#eff_BR:").strip()          # ...                      #
                    #print(l)                                   ############################
                    eff_dict[energy] = float(l)
                elif line.startswith('#efficiency:'):               
                    enbr = line.strip("#efficiency:").strip()          
                    eff_noBR_dict[energy] = float(enbr)
                elif line.startswith("#eff_err:"):
                    le = line.strip("#eff_err:").strip()
                    print (le)
                    eff_dict_err[energy] = float(le)

        #print ("eff_dict: ", eff_dict)
        #print ("eff_dict_err: ", eff_dict_err)
        efficiency_dict = {}
        for key in self.iso_dict:
            for idx in range(len(self.iso_dict[key][ke])):
                enrg = self.iso_dict[key][ke][idx]
                for key_e in eff_dict:
                    if enrg == key_e:
                        #efficiency_dict["Isotope",key,"Peak Energy",enrg] = "Efficiency",eff_dict[key_e],"Error",eff_dict_err[key_e]
                        if not key in efficiency_dict:
                          efficiency_dict[key] = {}
                        efficiency_dict[key][enrg] = (eff_dict[key_e], eff_dict[key_e]*eff_dict_err[key_e]/eff_noBR_dict[key_e])

        os.remove(_tmpfile)
        print ("... Efficiencies: ")
        for key in efficiency_dict:
          print (key, ": ", efficiency_dict[key])
        return efficiency_dict




    def simple_activity(self):
      ## first read in batfile and efficiencies
      pass
      









    def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):
      ## copied from stack overflow: https://stackoverflow.com/questions/14581358/getting-standard-errors-on-fitted-parameters-using-the-optimize-leastsq-method-i
      errfunc = lambda p, x, y: function(x,p) - y
      # Fit first time
      pfit, perr = optimize.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


      # Get the stdev of the residuals
      residuals = errfunc(pfit, datax, datay)
      sigma_res = np.std(residuals)

      sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

      # 100 random data sets are generated and fitted
      ps = []
      for i in range(100):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            optimize.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

      ps = np.array(ps)
      mean_pfit = np.mean(ps,0)

      # You can choose the confidence interval that you want for your
      # parameter estimates: 
      Nsigma = 1. # 1sigma gets approximately the same as methods above
                  # 1sigma corresponds to 68.3% confidence interval
                  # 2sigma corresponds to 95.44% confidence interval
      err_pfit = Nsigma * np.std(ps,0) 

      pfit_bootstrap = mean_pfit
      perr_bootstrap = err_pfit
      return pfit_bootstrap, perr_bootstrap 

      ## Usage:
      #pfit, perr = fit_bootstrap(pstart, xdata, ydata, ff)
      #print("\n# Fit parameters and parameter errors from bootstrap method :")
      #print("pfit = ", pfit)
      #print("perr = ", perr)



    def draw_lines_simple(self,
                   add_labels=True, 
                   add_fit=True,
                   xrange=None, 
                   ke='Peak Energies (keV)', 
                   klf='Lower Fit Range (keV)', 
                   kuf='Upper Fit Range (keV)',
                   col='k',
                   colBat="green",
                   do_count=False,
                   fontsize_result=7,
                   angle_result=90,
                   ):
        cax = plt.gca()
        _pos = 0.85 # to make sure 2 close dont overlap # Problem: it doesnt change after every energy line, it changes after every isotope in the isotope list
        _old_pos = 0.82
        _new_pos = 0.79
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        for key in self.iso_dict: 
            # key="Ti44","V48","Ra228", ...
            for idx in range(len(self.iso_dict[key][ke])): 
                # self.iso_dict= {"Ti44": {"Half-life (sec.)": [1893404160.0000002], "Number of peaks": [3.0], "Peak Energies (keV)": [67.9, 78.3, 1157.0], ...}, "V48": {...}, ...}
                enrg = self.iso_dict[key][ke][idx] # Peak Energies for Peak (idx) in every Isotope (key)
                lfr  = self.iso_dict[key][klf][idx] # Lower Fit Range for Peak (idx) in every Isotope (key)
                ufr  = self.iso_dict[key][kuf][idx] # Upper Fit Range for Peak (idx) in every Isotope (key)
                if xrange is None or (enrg > xrange[0] and enrg< xrange[1]):
                    #print (f'{key}: {enrg}')
                    cax.axvline(x=enrg, color=col, ls='--', lw=1) # vertical line at Peak Energies
                    cax.axvline(x=lfr, color=col, ls='--', lw=0.5) # vertical line at Lower Fit Range
                    cax.axvline(x=ufr, color=col, ls='--', lw=0.5) # vertical line at Upper Fit Range

                    trans = cax.get_xaxis_transform()
                    bat_result_str = ""

                    ### We now want to draw the BAT results. Hence we have to get the BAT fit results for the given 
                    ### isotope and multiply it by the BR and MC_efficiency and correct for the lifetime
                    if key in self.bat_vals:
                      _batr = self.bat_vals[key][0] # index 0 is the bat result; remaining indicees specify +/-unc and BF (maybe print?)
                      _batlerr = self.bat_vals[key][1] # index 0 is the bat result; remaining indicees specify +/-unc and BF (maybe print?)
                      _batuerr = self.bat_vals[key][2] # index 0 is the bat result; remaining indicees specify +/-unc and BF (maybe print?)
                      print ('BAT fit results=', _batr, ' for line@', enrg, ' keV (', key, ')' )
                      _effBR = self.eff_dict[key][enrg]
                      print ('  eff_BR: ', _effBR)
                      _times = self.get_times()
                      print (f'  realtime: {_times[0]/(3600*24):.2f} d, lifetime: {_times[1]/(3600*24):.2f} d' )
                      expected_counts = _batr * _effBR * _times[0]
                      print (f'  BAT expected counts at {enrg:.1f} keV: {expected_counts:.1f}')
                      ### Attention, by doing this we still ignore the uncertainty on the MC effBR determination!
                      expected_counts_lerr = (_batr-_batlerr) * _effBR * _times[0]
                      expected_counts_uerr = (_batr+_batuerr) * _effBR * _times[0]

                      _cts, _Xcts = self._integrate_range( xrange=(lfr,ufr) ) # range changeable (currecntly from -2...-1 and +1...+2)
#                    ## rough estimate of binwidth:
                      _bw = (self.x[_Xcts][1]-self.x[_Xcts][0])
                      _Dlu = ufr-lfr
#                    # flat bg_left:
#                    # integrate same interval width to the left
                      _cts_left, __Xctsl = self._integrate_range( xrange=(lfr-_Dlu, lfr) )
#                    # flag bg_right:
#                    # integrate same interval width to the right
                      _cts_right, __Xctsr = self._integrate_range( xrange=(ufr, ufr+_Dlu) )
#                    # average flat bg:
                      _av_bg = (_cts_left+_cts_right)/2.
                      _av_bg_err = 1/2. * np.sqrt(_av_bg)
                      _av_bg_rate = _av_bg/_Dlu * _bw 
                      _av_bg_rate_err = _av_bg_err/_Dlu * _bw
                      _cts_corr = _cts - _av_bg
                      _Drf = self.resolution_function(x=enrg)    

                      cax.fill_between(self.x[_Xcts], self.y[_Xcts], _av_bg_rate*np.ones_like(self.y[_Xcts]), facecolor='blue', alpha=0.5)
                      cax.hlines(y=_av_bg_rate, xmin=lfr-_Dlu, xmax=ufr+_Dlu, linestyles='-.', linewidths=1, colors=colBat)
                      _xgp0_bat, _ygp0_bat = self.draw_gauss_p0(A=expected_counts, mu=enrg, sig=_Drf, binwidth=_bw, p0=_av_bg_rate/_bw, color=colBat,verbosity=True)
                      _xgp0_bat_lerr, _ygp0_bat_lerr = self.draw_gauss_p0(A=expected_counts_lerr, mu=enrg, sig=_Drf, binwidth=_bw, p0=_av_bg_rate/_bw, color=colBat, verbosity=False, plot=False)
                      _xgp0_bat_uerr, _ygp0_bat_uerr = self.draw_gauss_p0(A=expected_counts_uerr, mu=enrg, sig=_Drf, binwidth=_bw, p0=_av_bg_rate/_bw, color=colBat, verbosity=False, plot=False)
                      ## TBC 
                      # indicate uncertainties on flat background determined by counting:
                      cax.fill_between([lfr-_Dlu, ufr+_Dlu], [_av_bg_rate+_av_bg_rate_err, _av_bg_rate+_av_bg_rate_err], [_av_bg_rate-_av_bg_rate_err, _av_bg_rate-_av_bg_rate_err], facecolor='red', alpha=0.3)
                      cax.fill_between(_xgp0_bat, _ygp0_bat_uerr, _ygp0_bat_lerr, facecolor="red", alpha=0.8, zorder=30)

#                     # Fit peak with gauss + flat bg:
                      res_fit_gauss_p0, cov_fit_gauss_p0 = self.fit_gauss_p0( xrange=(lfr-_Dlu, ufr+_Dlu), p0=(np.maximum(_cts_corr, 0), enrg, _Drf, _av_bg) )
                      _cts_fit = res_fit_gauss_p0[0]/_bw
                      _flat_bg_fit = res_fit_gauss_p0[1]/_bw
                      ncts_flat_bg_fit = _flat_bg_fit * _Dlu
                      error = [] 
                      for i in range(len(res_fit_gauss_p0)):
                        try:
                          error.append(np.absolute(cov_fit_gauss_p0[i][i])**0.5)
                        except:
                          error.append( 0.00 )
                      res_fit_gauss_p0_err = np.array(error)
                      print ("res_fit_gauss_p0_err = ", res_fit_gauss_p0_err)

                      ## Fit gauss + flat bg and bootstrap errors:
                      _pfit_bootstrap, _perr_bootstrap = fit_bootstrap(pstart, xdata, ydata, ff)
                      print("\n# Fit parameters and parameter errors from bootstrap method :")
                      print("_pfit_bootstrap = ", _pfit_bootstrap)
                      print("_perr_bootstrap = ", _perr_bootstrap)


                      bat_result_str = f"BAT: {expected_counts:.1f}-{expected_counts-expected_counts_lerr:.1f}+{expected_counts_uerr-expected_counts:.1f}; Fit: {_cts_fit:.1f}; Int: {_cts_corr:.1f}\n[p0: {ncts_flat_bg_fit:.1f}(fit); {_av_bg:.1f}(int)]"
#
                    if add_labels:
                        _t_pos = _old_pos # cycling between the 2 positions before each label
                        _old_pos = _pos
                        _pos = _new_pos
                        _new_pos = _t_pos

                        
                        if not do_count:
                            annotation_str = key

#                        
#                        cax.text(enrg, _pos, f'{annotation_str}', 
#                                 horizontalalignment='center', 
#                                 verticalalignment='center', 
#                                 transform=trans,
#                                 bbox=dict(facecolor='white', edgecolor='none'))

                        cax.text(lfr, 0.99, f'{annotation_str}', 
                                 horizontalalignment='right', 
                                 #verticalalignment='center', 
                                 verticalalignment='top', 
                                 transform=trans,
                                 rotation='vertical',
                                 bbox=dict(facecolor='none', edgecolor='none'))

                        y_top = 0.99
                        if key in self.bat_vals: 
                          #cax.text(ufr, y_top, f'{bat_result_str}', 
                          #       horizontalalignment='left', 
                          #       verticalalignment='top', 
                          #       transform=trans, 
                          #       color=colBat, 
                          #       #bbox=dict(facecolor='white', edgecolor='none'),
                          #       #bbox=dict(facecolor='none', edgecolor='none'),
                          #       fontsize=fontsize_result,
                          #       rotation='vertical',
                          #       )

                          cax.text(enrg, _av_bg_rate-2.*_av_bg_rate_err, f'{bat_result_str}', 
                                 horizontalalignment='center', 
                                 verticalalignment='top', 
                                 #transform=trans, 
                                 color=colBat, 
                                 bbox=dict(boxstyle='square,pad=0.1', facecolor='white', edgecolor='none'),
                                 #bbox=dict(facecolor='none', edgecolor='none'),
                                 fontsize=fontsize_result,
                                 rotation='horizontal',
                                 )

#                        if self.result_dict:
#                            #rate = result_dict[key]['rate background corrected'][idx]
#                            #rate_unc = result_dict[key]['rate background corrected uncertainty'][idx]
#                            _cfbg = self.result_dict[key]['compton background range'][idx]
#                            #print (f'{_cfbg}')
#                            cax.axvline(x=_cfbg[0], color=col, ls='-.', lw=0.3)
#                            cax.axvline(x=_cfbg[1], color=col, ls='-.', lw=0.3)
#
#                            cax.plot((_cfbg[2],_cfbg[4]), (_cfbg[3],_cfbg[5]), ls='--', color='m' )
#                            _cc = self.result_dict[key]['corrected counts'][idx]
#                            cax.text(enrg, _pos-0.15, f'{_cc[0]:.1f} +- {_cc[1]:.1f}cts', # lebel the lines
#                                     horizontalalignment='center', 
#                                     verticalalignment='center', 
#                                     transform=trans,
#                                     fontsize=8,
#                                     bbox=dict(facecolor='white', edgecolor='none')) # white box behind he label
#                            
                    if add_fit:
                      pass
#                        cax.fill_between(self.x[_Xcts], self.y[_Xcts],facecolor='blue', alpha=0.5)
#                        print (f'flat bg rate [{lfr-_Dlu:.1f}-{ufr+_Dlu:.1f}]keV: {_cts:.1f}cts ({_av_bg_rate:.1f}cts/keV)')
#                        print (f'              fit: {_cts_fit:.1f}cts ({_flat_bg_fit:.1f}cts/keV)')
#                        cax.hlines(y=_av_bg_rate, xmin=lfr-_Dlu, xmax=ufr+_Dlu, linestyles='-', linewidths=1, colors='m')
                      #  self.draw_gauss_p0(A=_cts_fit, mu=enrg, sig=_Drf, binwidth=_bw, p0=_flat_bg_fit, color='m' )






    def draw_lines(self, # Isotopes from the Isotope list
                   add_labels=True, 
                   add_fit=True,
                   xrange=None, 
                   ke='Peak Energies (keV)', 
                   klf='Lower Fit Range (keV)', 
                   kuf='Upper Fit Range (keV)',
                   col='k',
                   do_count=True,
                   ):
        cax = plt.gca()
        _pos = 0.85 # to make sure 2 close dont overlap # Problem: it doesnt change after every energy line, it changes after every isotope in the isotope list
        _old_pos = 0.82
        _new_pos = 0.79
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        for key in self.iso_dict: # key="Ti44","V48","Ra228", ...
            for idx in range(len(self.iso_dict[key][ke])): # self.iso_dict= {"Ti44": {"Half-life (sec.)": [1893404160.0000002], "Number of peaks": [3.0], "Peak Energies (keV)": [67.9, 78.3, 1157.0], ...}, "V48": {...}, ...}
                enrg = self.iso_dict[key][ke][idx] # Peak Energies for Peak (idx) in every Isotope (key)
                lfr  = self.iso_dict[key][klf][idx] # Lower Fit Range for Peak (idx) in every Isotope (key)
                ufr  = self.iso_dict[key][kuf][idx] # Upper Fit Range for Peak (idx) in every Isotope (key)
                if xrange is None or (enrg > xrange[0] and enrg< xrange[1]):
                    #print (f'{key}: {enrg}')
                    cax.axvline(x=enrg, color=col, ls='--', lw=1) # vertical line at Peak Energies
                    cax.axvline(x=lfr, color=col, ls='--', lw=0.5) # vertical line at Lower Fit Range
                    cax.axvline(x=ufr, color=col, ls='--', lw=0.5) # vertical line at Upper Fit Range

                    trans = cax.get_xaxis_transform()

                    _cts, _Xcts = self._integrate_range( xrange=(lfr,ufr) ) # range changeable (currecntly from -2...-1 and +1...+2)
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
                    _Drf = self.resolution_function(x=enrg)    
                    # Fit peak with gauss + flat bg:
                    res_fit_gauss_p0, cov_fit_gauss_p0 = self.fit_gauss_p0( xrange=(lfr-_Dlu, ufr+_Dlu), p0=(np.maximum(_cts_corr, 0), enrg, _Drf, _av_bg) )
                    _cts_fit = res_fit_gauss_p0[0]/_bw
                    _flat_bg_fit = res_fit_gauss_p0[1]/_bw

                    if add_labels:
                        _t_pos = _old_pos # cycling between the 2 positions before each label
                        _old_pos = _pos
                        _pos = _new_pos
                        _new_pos = _t_pos
                        
                        if not do_count:
                            annotation_str = key
                        else:
                            cax.fill_between(self.x[_Xcts], self.y[_Xcts],facecolor='blue', alpha=0.5)
                            annotation_str = f'{key} ({_cts:.0f}(bg:{_av_bg:.1f}) fit:{_cts_fit:.1f}(bg:{_flat_bg_fit*_Dlu:.1f}) cts)'
                        
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
                            cax.text(enrg, _pos-0.15, f'{_cc[0]:.1f} +- {_cc[1]:.1f}cts', # lebel the lines
                                     horizontalalignment='center', 
                                     verticalalignment='center', 
                                     transform=trans,
                                     fontsize=8,
                                     bbox=dict(facecolor='white', edgecolor='none')) # white box behind he label
                            
                    if add_fit:
                        cax.fill_between(self.x[_Xcts], self.y[_Xcts],facecolor='blue', alpha=0.5)
                        print (f'flat bg rate [{lfr-_Dlu:.1f}-{ufr+_Dlu:.1f}]keV: {_cts:.1f}cts ({_av_bg_rate:.1f}cts/keV)')
                        print (f'              fit: {_cts_fit:.1f}cts ({_flat_bg_fit:.1f}cts/keV)')
                        cax.hlines(y=_av_bg_rate, xmin=lfr-_Dlu, xmax=ufr+_Dlu, linestyles='-', linewidths=1, colors='m')
                        self.draw_gauss_p0(A=_cts_fit, mu=enrg, sig=_Drf, binwidth=_bw, p0=_flat_bg_fit, color='m' )


                            
    def draw_special_lines(self, sdict=None, add_labels=True, minBR=None, isotope=None, xrange=None, col='b', ax=None, angle=90, fontsize=8): # Isotopes from the manually added list
        if ax is None:
            cax = plt.gca()
        else:
            cax = ax
        _pos = 0.77 - 0.05
        _old_pos = 0.74 - 0.05
        
        if xrange is None:
            xrange = self._xlim
        else:
            self._xlim = xrange
        
        if sdict is None:
            sdict = self.manual_dict

        #if minBR is None:
        #    minBR=0
        for iso, infos in sdict.items():
            if isotope is not None:
              ### in case the isotopes-list is empty, we draw all lines **larger than _minBR**
              ### otherwise only the ones in the isotopes-list
              if not iso in isotope:
                continue
              elif isinstance(minBR, list) or isinstance(minBR, tuple):
                _minBR = minBR[ isotope.index(iso) ]
              elif minBR is None:
                _minBR=0
              else:
                ### we want to draw all lines larger than _minBR
                _minBR = minBR
            else:
              if minBR is None:
                _minBR=0
              else:
                _minBR=minBR
            for _inf in infos: # (_inf) = (Peak Energy, Branching Ratio)
                # print (_inf)
                try:
                    enrg, BR = _inf
                except Exception as e:
                    print (f'Exception: {e}\n_inf={_inf} (iso={iso})')
                if BR < _minBR:
                    continue

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
                        _pos = 0.01 #0.99
                        cax.text(enrg, _pos, 
                                 #f'{iso} (BR {BR})', 
                                 f'{iso} ({BR:.2f}%)', 
                                 horizontalalignment='left', 
                                 #verticalalignment='center', 
                                 verticalalignment='bottom',#'top', 
                                 transform=trans, 
                                 color=col, 
                                 bbox=dict(boxstyle='square,pad=0', facecolor='white', edgecolor='none'),
                                 #bbox=dict(facecolor='none', edgecolor='none'),
                                 fontsize=fontsize,
                                 rotation=angle,
                                 )


    def _get_nsig_intervals(self, mean, sigma, nlevels=3, x=None, y=None):
        if x is None:
            x=self.x
        if y is None:
            y=self.y
        ### let's count entries in +-levels*sigma around mean:
        intervals = np.arange(mean-nlevels*sigma, mean+(nlevels+1)*sigma, sigma) # np.arange(start,stop,step) -- evenly spaced values between start and stop with step size sigma
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
    
                        
    def create_pdf_summary(self, outfn=None, logy=False):
        xrange_full = np.arange(0,3600,150) # Creation of 23 equally sized ranges between 0 and 3450
        
        from matplotlib.backends.backend_pdf import PdfPages

        if outfn is None:
            outfn = f'{self.sample_name}_summary_raw.pdf'
            
        with PdfPages(outfn) as pdf: # to create multiple pdf
            for ir in range(len(xrange_full)-1):
                plt.figure(figsize=(12, 8)) # plotting a figure for every range

                xrange = xrange_full[ir:ir+2]
                title_str = f'{xrange[0]} keV - {xrange[1]} keV' # naming the plot with the energy range
                print (title_str)
                plt.title(title_str)


                # lw=linewidth, ls=linestyle, c=color
                plt.plot(self.x, self.y, lw=2, ls='-', c='k', label=f'{self.sample_name}') # plotting the data
                #plt.errorbar(self.x, self.y, yerr=np.sqrt(self.y), lw=2, ls='-', c='k', label=f'{self.sample_name}')

                self.draw_lines(add_labels=True, xrange=xrange) # drawing the lines in black
                #
                self.draw_special_lines(add_labels=True, xrange=xrange, col='m') # drawing the special lines in purple


                if not xrange is None:
                    plt.gca().set_xlim(*xrange)

                if logy:
                    plt.gca().set_yscale('log')
                plt.gca().set_xlabel('Energy / keV')
                plt.gca().set_ylabel('cts/bin')

                plt.legend()
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()

    def set_resolution_function(self, path_to_tf1_rootfile):
        ## execute get_calibration_function
        _tmpfn = "tmp.txt"
        exf1, exf2 = None, None
        _apppath = "/home/sebastian/Computing/GeMSE/GeMSE_root_scripts/GeMSE_ROOT_scripts/get_calibration_function"
        import subprocess
        with open(_tmpfn, "w") as f:
            #p1 = subprocess.run([_apppath, path_to_tf1_rootfile], stdout=f, text=True)
            p1 = subprocess.run(["singularity","exec",self._path_to_container_image,_apppath, path_to_tf1_rootfile], stdout=f, text=True)
        with open(_tmpfn, "r") as f:
            for line in f:
              w = line.split(":")
              if w[0]=="#formula":
                exf1=w[1].strip()
              elif w[0]=="#formulaP":
                exf2=w[1].strip()
              else:
                print("Odd format. I read this: '{line}'")
        os.remove(_tmpfn)

        if exf1 == "sqrt(([p0]+([p1]*x))+([p2]*(x*x)))":
          self.resolution_function = lambda x: eval( exf2.replace("sqrt","np.sqrt") )
          print (f"  Successfully set resolution_function. ({exf1}, {exf2})")
        else:
          print (f"Error in set_resolution_function. exf1={exf1}, exf2={exf2}")
        print (f"self.resolution_function(10)={self.resolution_function(10):.3f}")

#    def resolution_function(self, x):
        ## uproot seems not to be able to extract a TF1 from a TCanvas (as it is produced by GeMSE_analysis)
        ## let's do it manually for now...
#         import os
#         if self.bat_params is None:
#            #self.read_bat_params()
#            print ("You have to load a 'bat_params' file. Exiting. ")
#            return 0
#         _rf = self.bat_params["resolution file name"][0]
#
#         if not os.path.isfile(_rf):
#            print (_rf, ' does not exist.')
#            raise FileNotFoundError('File does not exist.')
#        
#         with uproot.open(_rf) as root_resolution:
#            hist = root_resolution["hist"]
#            h_edges = hist.axis().edges()
#            h_cont = np.array(list(hist.values()))

        ##########  
        # resolution function is: sqrt(([0] +([1]*x))+([2]*(x*x)))
        #return np.sqrt( (_respars[0] +(_respars[1]*x))+(_respars[2]*(x*x)) )
        ######
        # Please define when calling!
        ######
#        pass

    def gauss_p0(self, p0=None ):
        self.gauss_p0 = lambda x, p_0, p_3: p_0/np.sqrt(2*np.pi*p0[2]**2) * np.exp(-(x-p0[1])**2/(2*p0[2]**2) ) + p_3 # p0=amplitude, p0[1]=mu, p0[2]=sigma, p3=offset
        # TBC



    def fit_gauss_p0(self, xrange, p0=None ):
        #  xrange = (lfr-_Dlu, ufr+_Dlu)
        #  p0 = (_cts_corr, enrg, _Drf, _av_bg)
        from scipy.optimize import curve_fit
        ## Attention! number of counts cts = A*binwidth
        # _func = lambda x, p0, p1, p2, p3: p0/np.sqrt(2*np.pi*p2**2) * np.exp(-(x-p1)**2/(2*p2**2) ) + p3
        _func_constr = lambda x, p_0, p_3: p_0/np.sqrt(2*np.pi*p0[2]**2) * np.exp(-(x-p0[1])**2/(2*p0[2]**2) ) + p_3 # p0=amplitude, p0[1]=mu, p0[2]=sigma, p3=offset
        _Xr = ((self.x>xrange[0]) & (self.x<xrange[1]))
        _popt, _pcov = curve_fit(_func_constr, self.x[_Xr], self.y[_Xr], p0=(p0[0],p0[3]), bounds=((0,0),(np.inf,np.inf))  )
        return _popt, _pcov


    def fit_spectrum(self, ke='Peak Energies (keV)', klf='Lower Fit Range (keV)', kuf='Upper Fit Range (keV)',):
        _sample_dict = {}
        #_d = {}
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
                _cts_fit = round(res_fit_gauss_p0[0]/_bw, 6)
                _cts_delta = round(np.sqrt(cov_fit_gauss_p0[0][0]/_bw), 6)
                _flat_bg_fit = res_fit_gauss_p0[1]/_bw

                #print (f'flat bg rate [{lfr-_Dlu:.1f}-{ufr+_Dlu:.1f}]keV: {_cts:.1f}cts ({_av_bg_rate:.1f}cts/keV)')
                #print (f'              fit: {_cts_fit:.1f}cts ({_flat_bg_fit:.1f}cts/keV)')
                #_d["Counts"] = _cts_fit
                _sample_dict["Isotope",key,"Peak Energy",enrg] = "Counts",_cts_fit,"Error",_cts_delta
        return _sample_dict

    def save_peaks2(self, _type="bkg"):
        _sample_dict = self.fit_spectrum()
        if _type=="bkg":
            fn=self.bkgdict_txt
        elif _type=="sample":
            fn=self.sampledict_txt
        with open(fn, "w") as f:
            str_key = repr(_sample_dict)
            f.write(str_key)


    def get_times(self):
        _tmpfn = "times.txt"
        _s = []
        _apppath = "/home/sebastian/Computing/GeMSE/GeMSE_root_scripts/GeMSE_ROOT_scripts/get_times_from_rootfile"
        import subprocess
        with open(_tmpfn, "w") as f:
            #p1 = subprocess.run([_apppath, self.fn], stdout=f, text=True)
            p1 = subprocess.run(["singularity","exec",self._path_to_container_image,_apppath, self.fn], stdout=f, text=True)
        with open(_tmpfn, "r") as f:
            for line in f:
              if not line.startswith('# '):
                _s.append(float(line))
        os.remove(_tmpfn)
        return _s
 

    def get_activity(self):
        # Calculation of the meassuretime corrected rates
        import os
        _s = []
        _b = []                                       # sample-/bkgtimes.txt looks like
        with open(self.sampletimes_txt, "rt") as f:      ###################
            for line in f:                            # "# t_real in s" #
                if not line.startswith('# '):         # t_real          #
                    _s.append(float(line))            # "# t_live in s" #
            f.close()                                 # t_live          #
        with open(self.bkgtimes_txt, "rt") as f:         ###################
            for line in f:                            
               if not line.startswith('# '):          
                    _b.append(float(line))            
            f.close()  
        #os.remove("output.txt") # delete txt         
        t_sample_real = _s[0]/3600/24 # in hours
        t_sample_live = _s[1]/3600/24 # in hours
        t_bkg_real = _b[0]/3600/24 # in hours
        t_bkg_live = _b[1]/3600/24 # in hours

        #t_sample_real = 24.835532
        #t_sample_live = 24.835532
        #t_bkg_real = 109.726273
        #t_bkg_live = 109.726157
        t_sample = min(t_sample_real,t_sample_live)
        t_bkg = min(t_bkg_real,t_bkg_live)
        print("sample measurement time:",round(t_sample,1), "hours, background measurement time:",round(t_bkg,1),"hours")

        with open(self.sampledict_txt, "r") as f:
            for line in f:
                sample_dict = line
        with open(self.bkgdict_txt, "r") as f:
            for line in f:
                background_dict = line
        sample_dict = eval(sample_dict)
        background_dict = eval(background_dict)
        print(sample_dict)
        print(background_dict)
        #print(type(sample_dict))
        #print(type(background_dict))
        rate_dict = {}
        for (key_s,key_b) in zip(sample_dict, background_dict):
            rate = sample_dict[key_s][1]/t_sample - background_dict[key_b][1]/t_bkg
            rate_err = np.sqrt((sample_dict[key_s][3]/t_sample)**2 - (background_dict[key_b][3]/t_bkg)**2)
            #print("Rate:",key_s[1],"at",key_s[3],"keV: (",round(rate,1),"pm",round(rate_err,1),")/day")
            rate_dict[key_s] = "rate",rate,"error",rate_err     # key_s = "Isotope",key,"Peak Energy",enrg
        print(rate_dict)

        # Calculation of the activities of the isotopes
        eff_dict = self.get_efficiencies()
        print(eff_dict)
        activity_dict = {}
        for (key_r,key_e) in zip(rate_dict, eff_dict):
            activity = rate_dict[key_r][1]/eff_dict[key_e][1]/3600/24
            activity_err = rate_dict[key_r][3]/eff_dict[key_e][1]/3600/24
            print("Activity:",key_r[1],"at",key_r[3],"keV: (",round(activity,3),"pm",round(activity_err,3),")Bq")
            activity_dict[key_r] = "activity",activity,"error",activity_err     # key_r = "Isotope",key,"Peak Energy",enrg

        # Calculation of the total activity of one Isotope
        d = []
        for key_r in activity_dict:
            d.append(key_r[1])
            isotope = list(dict.fromkeys(d))
        #print(isotope)

        for i in isotope:
            tot_activity = 0
            tot_activity_err = 0
            error_list = []
            for key_r in activity_dict:
                if key_r[1] == i:
                    tot_activity = tot_activity + activity_dict[key_r][1]
                    error_list.append(activity_dict[key_r][3])
            for n in range(len(error_list)):
                tot_activity_err = tot_activity_err + error_list[n]**2
            tot_activity_err = np.sqrt(tot_activity_err)
            print("Total activity of",i,":(",round(tot_activity,3),"pm",round(tot_activity_err,3),")Bq")
            print("                   ({:.1e} pm {:.1e}) mBq".format(tot_activity*1e3, tot_activity_err*1e3))
            print("                   ({:.2f} pm {:.2f}) mBq".format(tot_activity*1e3, tot_activity_err*1e3))



    def draw_gauss_p0(self, A, mu, sig, binwidth=None, nsig=5, p0=None, color='r',verbosity=True, plot=True):
        if verbosity:
          try:
            print (f"calling draw_gauss_p0 using...\n\tA={A:.1f}, mu={mu:.1f}, sig={sig:.1f},\n\tbinwidth={binwidth}, nsig={nsig}, p0={p0}")
          except:
            print(f"calling draw_gauss_p0 using...\n\tA={A}, mu={mu}, sig={sig},\n\tbinwidth={binwidth}, nsig={nsig}, p0={p0}")
        cax = plt.gca()
        _x = np.linspace(mu-nsig*sig, mu+nsig*sig, num=100)
        _y2 = A/np.sqrt(2*np.pi*sig**2) * np.exp(-(_x-mu)**2/(2*sig**2) )
        if not p0 is None:
          _y2 += p0 # incase of a offset
        ## account for binning:
        if not binwidth is None:
            # default binwidth=1
            _y2 = _y2*binwidth # incase of a binwidth
        if plot:
          cax.plot(_x, _y2, color=color, ls='--', lw=2) # dashed line
        return _x, _y2



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

                            ## now trying to draw bat result on top of flat background 

                            if key in self.bat_vals:
                              _A_bat = self.bat_vals[key][0]
                              print (f'Bat result for {key} = {_A_bat} Bq')


                              _N_bat = _A_bat * self.get_times[1] 

                              self.draw_gauss_p0(A=_A_bat/_bw, mu=enrg, sig=_Drf, binwidth=_bw, p0=_av_bg_rate/_bw, color='g' )
                            else:
                              print (f'{key} not in {self.bat_vals}...')

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




    def create_pdf_summary_bat2(self, outfn=None, stepsize=150, Emin=0, Emax=3600, logy=True, errorbar=False):
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
                self.draw_lines_simple(add_labels=True, xrange=xrange, col="k", colBat="red")
                self.draw_special_lines(add_labels=True, xrange=xrange, col='m', angle=90, fontsize=8) # drawing the special lines in purple
                
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

    def create_pdf_summary_bat(self, outfn=None, stepsize=150, Emin=0, Emax=3600, logy=True, errorbar=False, do_count=True):
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
                title_str = f'{xrange[0]} keV - {xrange[1]} keV   --  {self.fn}'
                print (title_str)
                plt.title(title_str)

                if errorbar:
                  # schoener, aber viel langsamer:
                  plt.errorbar(self.x, self.y, yerr=np.sqrt(self.y), lw=2, ls='-', c='k', label=f'{self.sample_name}')
                else:
                  plt.plot(self.x, self.y, lw=2, ls='-', c='k', label=f'{self.sample_name}')

                ## TODO 
                # self.draw_lines(add_labels=True, xrange=xrange)
                self.draw_lines_bat(add_labels=True, xrange=xrange, do_count=do_count)
                #
                ## 
                # TODO self.draw_special_lines(add_labels=True, xrange=xrange, col='m')
                #
                self.draw_special_lines(add_labels=True, xrange=xrange, col='m') # drawing the special lines in purple




                if not xrange is None:
                    plt.gca().set_xlim(*xrange)
                #plt.gca().set_ylim(0.1, None)
                
                if logy:
                    plt.gca().set_yscale('log')
                plt.gca().set_xlabel('Energy / keV')
                plt.gca().set_ylabel('cts/bin')

                plt.legend()
                plt.tight_layout()
                pdf.savefig()  # saves the current figure into a pdf page
                plt.close()
