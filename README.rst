GemsePyAna
--------

To install, clone repo and (inside your favorite conda environment) do::

    >>> pip install -e .


To use in a script, do something along the lines::


    >>> from gemsepyana import GeMSEData, simple_activities, plot_actvs, which_chain

Then load the data::

    >>> al = GeMSEData()  # the investigated sample
    >>> al.fn = '[path_to_rootfiles]/combined_spectra_runs__1_2.root'
    >>> al.sample_name = r'Short catchy name'
    >>> al.t_live = 157987+346328 # in seconds
    >>> al.load_spectrum()
    >>> al.rebin(nbins=10) # often looks nicer


    >>> bg = GeMSEData() # similar for background files
    >>> bg.fn = '[path_to_bg_rootfile]/2019-2020_combined_bkg_calibrated.root.root'
    >>> bg.sample_name = 'Background (2020)'
    >>> bg.t_live = 9.48034e+06 # in seconds
    >>> bg.load_spectrum()
    >>> bg.rebin(nbins=10)

And e.g. display it::

    >>> c_al = 'xkcd:poop green'
    >>> c_bg = 'xkcd:royal blue'

    >>> fig, ax = plt.subplots(figsize=(16,8))

    >>> xrange = (0,3000)
    >>> isotope = ("U238", "Th228", "Tl208")  # can be None or any list
    >>> minBR = 0 # show only lines with BR>minBR

    >>> ax.plot(al.x , al.y_per_keV_per_day , '-', c=c_al1, lw=1, label=al.sample_name)
    >>> ax.plot(bg.x, bg.y_per_keV_per_day, '-', c=c_bg, lw=1, label=bg.sample_name)

    >>> if True:
    >>>     al.draw_special_lines(
    >>>         sdict = al.manual_dict,
    >>>         col = 'k',
    >>>         xrange=xrange,
    >>>         minBR = minBR,
    >>>         isotope=isotope,
    >>>         )

    >>> ax.set_ylabel(r'Counts [keV$^{-1}$ day$^{-1}$]')
    >>> ax.set_xlabel('Energy [keV]')
    >>> ax.set_yscale('log')
    >>> ax.set_xlim(*xrange)
    >>> ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2),
    >>>           loc="lower left",
    >>>           mode="expand",
    >>>           borderaxespad=0,
    >>>           frameon=False,
    >>>           ncol=2)

    >>> fig.show()

A simple rate analyis could look like::

   >>> al.load_bat_params(fn="[path_to_GeMSE_analysis_type]/parameters_activity_calculation.txt")
   >>> actvs, als = simple_activities(gd=al, plot=True, isotopes=["Th228",], yscale='linear') # compute individual rates per line

   >>> plot_actvs(als=als) # plot rates of different lines belonging to one isotope


More examples to be provided here::


Packaging of files based on::

    https://python-packaging.readthedocs.io/en/latest/
