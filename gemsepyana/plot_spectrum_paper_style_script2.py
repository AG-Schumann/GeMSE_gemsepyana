import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
import sys
import getopt
import os
import re
from . import GeMSEData

plt.style.use( os.path.join( os.path.dirname(__file__), 'gemse.mplstyle') )

print ('Environments set, classes loaded. Processing data...')

def usage():
  print (f'{sys.argv[0]} -i <inputfile> -b <backgroundfile> -o <outputfile> [--isotopes="U238,Ra226_0.8,Tl208"]')


def main(argv=sys.argv):

  c_sample='k'
  c_bg = 'g'

  inputfile = ''
  bgfile = ''
  outputfile = ''
  isotopes_str = ''
  try:
    opts, args = getopt.getopt(argv[1:],"hi:b:o:",["ifile=","bgfile=","ofile=", "isotopes="])
  except getopt.GetoptError:
    usage()
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       usage()
       sys.exit()
    elif opt in ("-i", "--ifile"):
       inputfile = arg 
    elif opt in ("-b", "--bgfile"):
       bgfile = arg 
    elif opt in ("-o", "--ofile"):
       outputfile = arg 
    elif opt in ("--isotopes"):
       isotopes_str = arg
       print ("Found isotopes: ", isotopes_str)

  isotope = None
  BR = None
  if isotopes_str:
    isotope=[]
    BR=[]
    for isoBR in isotopes_str.split(','):
      ib = isoBR.split('_')
      isotope.append(ib[0])
      if len(ib)==2:
        BR.append(float(ib[1]))
      else:
        BR.append(0.0)

    if len(isotope)==1 and isotope[0]=='all':
        isotope=None
        BR = BR[0]


  # sample
  al = GeMSEData()
  al.fn = inputfile

  al.sample_name = os.path.basename(outputfile).rstrip('_paperStylePlot.pdf')

  #al.t_live = int( os.path.basename(al.fn).split('.root')[0].split("_")[-1].rstrip('s') )
  _pattern = r'\d{3,}s' # three or more digits followed by the letter s
  _times = re.findall( _pattern, os.path.basename(al.fn) )
  try:
    al.t_live = int(_times[-1].strip('s'))
  except:
    print ("")
    print (f"ERROR. Regex can't match a duration in filename? fn={os.path.basename(al.fn)}, _times={_times}")
    print (f"       Will set duration to t=1s for now. Rates won't make sense!!! Need to check!")
    print ("")
    al.t_live=1

  _nw = 25 
  if len(al.sample_name)>_nw:
    _short_name = al.sample_name.split("_combined")[0]
    if len(_short_name)>_nw:
      _short_name = _short_name[:_nw]
    _short_name += "..."
  else:
    _short_name = al.sample_name

  #al.sample_name += f" ({al.t_live/(3600*24):.1f} days)"
  al.sample_name = f"{_short_name} ({al.t_live/(3600*24):.1f} days)"

  al.load_spectrum()
  al.rebin(nbins=5)


  # bg file
  bg = GeMSEData()
  bg.fn = bgfile
  #bg.sample_name = 'Background (2020)'
  bg.sample_name = os.path.basename(bg.fn).split('.root')[0]
  bg.t_live = 9.48034e+06 # in seconds

  bg.load_spectrum()
  bg.rebin(nbins=5)

  fig, ax = plt.subplots(figsize=(16,10))

  xrange = (0,3000)
  ax.plot(al.x , al.y_per_keV_per_day , '-', c=c_sample, lw=1, label=al.sample_name)
  ax.plot(bg.x, bg.y_per_keV_per_day, '-', c=c_bg, lw=1, label=bg.sample_name)

  print (f"Drawing lines with isotopes={isotope} and minBR={BR}")

  ax.set_ylabel(r'Counts [keV$^{-1}$ day$^{-1}$]')
  ax.set_xlabel('Energy [keV]')
  ax.set_yscale('log')
  ax.set_xlim(*xrange)
  ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), 
            loc="lower left",
            mode="expand", 
            borderaxespad=0,
            frameon=False, 
            ncol=2)

  annots = al.annotate_lines(
        sdict = al.manual_dict, ## put cuts on manual dict, to select important lines
        xrange=xrange,
        minBR = BR,
        isotope=isotope,
        fontsize=10,
        adjustTextLabels=False,
        #padding=100,
        #pixel_offset = (0, 50),
        #drawVLines=False, 
  ) 


  if True:
    adjust_text( annots, ax=plt.gca() ) #, only_move={'text':'y+'} )  ## this is an interesting feature...
  #fig.show()
  plt.savefig(outputfile)


##########################################################
## Similar to "create_pdf_summary", produce zooms:
  xrange_full = np.arange(0,3600,150)
  from matplotlib.backends.backend_pdf import PdfPages
  
  outfn = f'{outputfile}_zooms.pdf'

  bg.rebin(nbins=1)
  al.rebin(nbins=1)

  with PdfPages(outfn) as pdf:
    #plt.style.use('/home/sebastian/.pltstyle/gemse_small.mplstyle') 
    for ir in range(len(xrange_full)-1):
      plt.figure(figsize=(16, 10))
      xrange = xrange_full[ir:ir+2]
      title_str = f'{xrange[0]} keV - {xrange[1]} keV'
      print (title_str)

      plt.plot(al.x , al.y_per_keV_per_day , ls='-', c=c_sample, lw=2, label=al.sample_name)
      plt.plot(bg.x, bg.y_per_keV_per_day, ls='-', c=c_bg, lw=2, label=bg.sample_name)

      if not xrange is None:
        plt.gca().set_xlim(*xrange)
        plt.gca().set_yscale('log')
        plt.gca().set_xlabel('Energy / keV')
        plt.gca().set_ylabel(r'Counts [keV$^{-1}$ day$^{-1}$]')
        plt.gca().legend(bbox_to_anchor=(0, 1.02, 1, 0.2), 
            loc="lower left",
            mode="expand", 
            borderaxespad=0,
            frameon=False, 
            ncol=2)

      plt.tight_layout()


      annots = al.annotate_lines(
            sdict = al.manual_dict, ## put cuts on manual dict, to select important lines
            xrange=xrange,
            minBR = 0.0, # BR,
            isotope= None, # isotope,
            fontsize=10,
            adjustTextLabels=False,
            #padding=100,
            #pixel_offset = (0, 50),
            #drawVLines=False, 
      ) 

      if True:
        adjust_text( annots, ax=plt.gca() ) #, only_move={'text':'y+'} )  ## this is an interesting feature...

      pdf.savefig()  # saves the current figure into a pdf page
      plt.close()
##########################################################





if __name__ == "__main__":
   main(sys.argv)


