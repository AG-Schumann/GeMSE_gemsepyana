import matplotlib.pyplot as plt
import uproot
import numpy as np
import sys
import getopt

#from analysis_classes.gemsedata_uproot import GeMSEData_uproot as GeMSEData
from . import GeMSEData2

print ('Environments set, classes loaded. Processing data...')

def usage():
  print (f'{sys.argv[0]} -i <inputfile> -o <outputfile>')

def main(argv=sys.argv):
   #print (f"inside main. argv={argv}")
   inputfile = ''
   outputfile = ''
   try:
      opts, args = getopt.getopt(argv[1:],"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      usage()
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         usage()
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg


   fn = inputfile
   sample = GeMSEData2(fn=fn)
   
   sample.load_spectrum()
   # sample.plot_spectrum(fn=outputfile)

   sample.create_pdf_summary(outfn=outputfile)

   print (f'Successfully turned {inputfile} into {outputfile}.')

if __name__ == "__main__":
   main(sys.argv)



