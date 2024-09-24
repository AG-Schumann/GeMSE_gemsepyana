import matplotlib.pyplot as plt
import uproot
import numpy as np
import sys, getopt

#from analysis_classes.gemsedata_uproot import GeMSEData_uproot as GeMSEData
from . import GeMSEData2
print ('Environments set, classes loaded. Processing data...')

def usage():
  print (f'{sys.argv[0]} -p <bat_params> -i <inputfile> -o <outputfile>')

def main(argv=sys.argv):
  inputfile = ''
  outputfile = ''
  try:
    opts, args = getopt.getopt(argv[1:],"hp:i:o:",["batfile=", "ifile=","ofile="])
  except getopt.GetoptError:
    usage()
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       usage()
       sys.exit()
    elif opt in ("-p", "--batfile"):
       batparamsfile = arg
    elif opt in ("-i", "--ifile"):
       inputfile = arg
    elif opt in ("-o", "--ofile"):
       outputfile = arg


  fn = inputfile 
  sample = GeMSEData(fn=fn)

  bat_pars = sample.read_bat_params(fn=batparamsfile)
  #for k in bat_pars.keys():
  #  print (f"{k}: {bat_pars[k]}")
  #return 0

#### unfortunately up to now these numbers have to be manually extracted... ###
#### TODO write simple c++ script to do this automatically ####
  ## resolution function is sqrt(([0] +([1]*x))+([2]*(x*x)))
  sample._respars = [0.2392, 0.0002828, 2.327e-8]
  sample._resparserr = [0.007104, 1.915e-5, 8.212e-9] 
  sample.resolution_function = lambda x: np.sqrt((sample._respars[0] +(sample._respars[1]*x))+(sample._respars[2]*(x*x)))
  ## cal funtion is pol2: [0] + [1]*x + [2]*x^2
  sample._calpars = [0.1904, 0.09928, 1.087e-9]
  sample._calparserr = [0.02739, 5.977e-6, 3.058e-10]

  sample.load_spectrum()

  do_simulate_spectrum = True
  if do_simulate_spectrum:
    sample.fit_dict = {}
    enrg = []
    ke='Peak Energies (keV)'
    signal_avg = 150 # total counts per peak
    signal_spread = 5 # spread of total counts per peak
    flat_bg_avg = 15 # cts/keV
    flat_bg_spread = 3 # spread of flat background
    for key in sample.iso_dict:
      for idx in range(len(sample.iso_dict[key][ke])):
        _enrg = sample.iso_dict[key][ke][idx] 
        enrg.append ( float(_enrg) )

    enrg = np.array ( enrg )
    sample.fit_dict['energy'] = enrg
    sample.fit_dict['isotope'] = sample.iso_dict

    from numpy.random import uniform
    from numpy.random import normal

    sample.fit_dict['signal_true'] = normal(loc=signal_avg, scale=signal_spread, size=len(enrg) ) # in absolutes counts per peak
    sample.fit_dict['flat_bg_true'] = normal(loc=flat_bg_avg, scale=flat_bg_spread, size=len(enrg) ) # in cts/keV
    sample.fit_dict['signal_fit'] = None
    sample.fit_dict['flat_bg_fit'] = None

    #cts = ncts*np.ones_like( enrg )
    #res = sample.resolution_function( enrg )
    sample.simulate_spectrum(
        cts=sample.fit_dict['signal_true'], 
        enrg=sample.fit_dict['energy'], 
        res=sample.resolution_function( sample.fit_dict['energy'] ) , 
        xrange=(0,3600), 
        flat_bg=sample.fit_dict['flat_bg_true'], 
        thrl=10, 
        thrh=3100,
    )
    # overwrite y data of loaded_spectrum to make plotting easier... (TODO make nice)
    sample.y = sample.y_sim

  
  #sample.plot_spectrum(fn=outputfile)

  sample.fit_spectrum()
  
  # print (f'self.iso_dict = {sample.iso_dict}')
  sample.create_pdf_summary_bat(outfn=outputfile, Emin=0, Emax=3500, stepsize=75, logy=False, errorbar=False)

  print (f'Successfully created {outputfile} from {inputfile}.')

if __name__ == "__main__":
   main(sys.argv)



