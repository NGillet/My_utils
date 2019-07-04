##############################################################################
##############################################################################
### A bunch of more or less usefull functions functions 
##############################################################################
##############################################################################

import numpy as np
from time import sleep, time ### loadbar
import sys                   ### loadbar

from ipywidgets import IntProgress, HTML, VBox ### progressBar
from IPython.display import display            ### progressBar

import pyDOE 
from scipy.interpolate import interp1d
from scipy.interpolate import griddata

import pymc3 as pm

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator, ScalarFormatter
import matplotlib.cm as cm

##############################################################################
def zcen(bins, log=False):
    """
    Return the center bin edges array.
    
    Note
    ----
    Usefull for histogram plots.

    Parameters
    ----------
    bins_edges : array_like[nbins_edges,]
       The bins edges.
    log : Optional[bool]
        If the bins edges are in lin but log spaced, the log flag should be put at True to obtain the log center.
    
    Returns
    -------
    bins_center : array_like[nbins_edges-1,]
        The center of the bins.
    
    Raises
    ------
    None
    """
    if log :
        return 10**(np.log10(bins[:-1])+np.diff(np.log10(bins))/2)
    else:
        return bins[:-1] + np.diff(bins)/2.
    
##############################################################################
def loadbar( iterator, Ntot, tin, ifJump=False, com='' ):        
    """
    Print a loading-bar in a for loop
    
    Note
    ----
    - need to: import time.time as time
    /!\ if you already have an onther "time" library (like EMMA time)
    - if your iteration is too fast, it might slow it down !

    Parameters
    ----------
    iterator : int
        The number of the curent iteration.
    Ntot : int
        The total number of iteration. 
    tin  : float
        The time at the begening of the for-loop /!\ initialize as tin=time() before the for-loop
    ifJump : Optional[bool]
        Go to the line at each iteration. Not recomended if there Ntot is large.
    com = Optional[str('')]
        Add a commentaire at each iteration.
    
    Returns
    -------
    bins_center : array_like[nbins_edges-1,]
        The center of the bins.
    
    Raises
    ------
    None
    """
    titerAvg = (time() - tin) / (iterator+1)
    
    sys.stdout.write( com )
    sys.stdout.write('\r')
    sys.stdout.write('\r')
    if( ifJump and not(iterator) ):
        sys.stdout.write('\n')
    sys.stdout.write( "[%-20s] %.1f%%"%('='*int((iterator+1)*100./Ntot/5),(iterator+1)*100./Ntot) )
    sys.stdout.write( ", %.1f s/iter, remain %.1f s, exec %.1f s"%( titerAvg, titerAvg*(Ntot-iterator-1), time()-tin ) )    
    sys.stdout.flush()
    #return time()

##############################################################################
def find_idx_nearest_val( array, value ):
    """
    Return the id of the element of 'array' which is the closest to 'value'.
    
    Note
    ----
    - Usefull to find the closest snapshot ID to a given redshift.
    - array is sorted, the result might not be unique so.

    Parameters
    ----------
    array : array_like[nsamples,]
        The array of values to find the closest ID in. 
    value : same type as array
        The value you want to find in array (the closest one).
    
    Returns
    -------
    idx_nearest : int 
        The id of array that correspond to value (the closest one).
    
    Raises
    ------
    None
    
    TODO
    ----
    Make it vectorise on 'value', for the moment it need a for loop if you have several 'values'.
    """
    idx_sorted = np.argsort(array)
    sorted_array = np.array(array[idx_sorted])
    idx = np.searchsorted(sorted_array, value, side="left")
    ### first the 2 extrem cases
    if idx >= len(array):
        idx_nearest = idx_sorted[len(array)-1]
    elif idx == 0:
        idx_nearest = idx_sorted[0]
    ### in a middle case
    else:
        ### find the closest between the larger or smaller
        if abs(value - sorted_array[idx-1]) < abs(value - sorted_array[idx]):
            idx_nearest = idx_sorted[idx-1]
        ### if they two (minimum) identicals
        else:
            idx_nearest = idx_sorted[idx]
    return idx_nearest

##############################################################################
def sigma123_2D( proba2D, confidence_levels=[0.68, 0.95, 0.997] ):
    """
    Return the value of the probabilities where it is 1, 2 and 3 sigmas (68%, 95%, 99.7% confidence level).
    For a 2D pdf.
    
    Note
    ----
    - Usefull to plot the 1,2,3 sigmas with 2D contours.
    - It is a HPD (Highest Posterior Density): 
    sort and sum the proba from the largest one until it reach the desire confidence levels.

    Parameters
    ----------
    proba2D : array_like[nX,Ny,] (2D)
        The 2D pdf: proba2D.sum() == 1
    confidence_levels : Optional[array_like[3,]]
        The three confidence level to probe.
    
    Returns
    -------
    proba_at_CL : array_like[3,]
        The three probability corresponding to the confidence level. 
    
    Raises
    ------
    None
    
    TODO
    ----
    - change the name - this function is general work for NDim proba !
    
    see also
    --------
    find_confidence_interval : which gives the same result
    find_proba_limit : which gives the same result
    """
    proba2D_sort = np.argsort(proba2D.flatten())[::-1]
    p1 = proba2D.flatten()[proba2D_sort][ np.where( np.cumsum( proba2D.flatten()[proba2D_sort] ) > confidence_levels[0] )[0][0] ]
    p2 = proba2D.flatten()[proba2D_sort][ np.where( np.cumsum( proba2D.flatten()[proba2D_sort] ) > confidence_levels[1] )[0][0] ]
    p3 = proba2D.flatten()[proba2D_sort][ np.where( np.cumsum( proba2D.flatten()[proba2D_sort] ) > confidence_levels[2] )[0][0] ]
    
    return p1, p2, p3

##############################################################################

import scipy.optimize as so
def find_confidence_interval( x, pdf, confidence_level ):
    """
    Return the value of the probabilitie at the confidence_level.
    
    Note
    ----
    - This function is maint to be use with import scipy.optimize as so : 
    
    Example
    -------
    ONE_sigma = so.brentq( find_confidence_interval, 0., 1., args=(proba,0.68) ) 

    Parameters
    ----------
    x : 
        variable for so.brentq
    pdf : array_like[NDim,]
        The NDim pdf
    confidence_level : float inside [0,1]
        the confidence level to reach
    
    Returns
    -------
    proba_at_CL : float
        The probability corresponding to the confidence level. 
    
    Raises
    ------
    None
    
    see also
    --------
    sigma123_2D : which gives the same result
    find_proba_limit : which gives the same result
    """
    return pdf[pdf>x].sum() - confidence_level

##############################################################################
def find_proba_limit( pdf, confidence_level=0.68 ):
    """
    Return the value of the probabilitie at the confidence_level.
    
    Note
    ----

    Parameters
    ----------
    pdf : array_like[NDim,]
        The NDim pdf
    confidence_level : float inside [0,1]
        the confidence level to reach
    
    Returns
    -------
    proba_at_CL : float
        The probability corresponding to the confidence level. 
    
    Raises
    ------
    None
    
    see also
    --------
    sigma123_2D : which gives the same result
    find_confidence_interval : which gives the same result
    """
    
    pdf = pdf.ravel()
    
    ID_sorted_pdf = np.argsort( pdf )[::-1]
        
    ### when there are some point that match exactly the confidence_level
    ### return the proba value of the first of them
    if (pdf[ID_sorted_pdf].cumsum()==confidence_level).sum() :
        ID_p1 = ID_sorted_pdf[ (pdf[ID_sorted_pdf].cumsum()==confidence_level) ][0]
        p1 = pdf[ID_p1]
    else:
        ### take all ID of values above confidence_level
        ze = np.where( ( (pdf[ID_sorted_pdf].cumsum())>confidence_level ) )[0][0]
        ### if it is the first element
        if ze==0:
            ze=1
        ### interpolate between the two closest values
        idp = ID_sorted_pdf[ze]
        idm = ID_sorted_pdf[ze-1]
        p1 = np.interp( confidence_level, 
                   [pdf[ID_sorted_pdf].cumsum()[ze-1],pdf[ID_sorted_pdf].cumsum()[ze]], 
                   [pdf[idm],pdf[idp]] )
    return p1

##############################################################################      
def saveFig( figs, name ):
    """
    save a figure in a pdf file
    I don't know why I made this. I must have copy it from somewhere (SO)
    """
    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages( name )
    try:
        Nfig = len(figs)
        for f in figs:
            pp.savefig( f, bbox_inches='tight' )
    except TypeError:
        pp.savefig( figs, bbox_inches='tight' )
    pp.close()

##############################################################################
def compute_stat( list_points, weights, Nbin=200, rangeBin=None, DoInterp=True, DoHisto=False ):
    """
    Compute the marginalized proba 1D and 2D for N-dimensions 
    - with an interpolation on a grid option
    - or simple histogram option
    
    Note
    ----
    - Main fuction for the triangular plot (c.f. plot_triangular) 
    
    Parameters
    ----------
    list_points : array_like[nPoints,NDIM,]
        Contain the sample of points.
    weights : array_like[nPoints,]
        The weights of each points.
    Nbin : Optional[int]
        Dimention of the final grid.
    DoInterp : Optional[bool]
        Interpolate the weights on a grid of dimention power(Nbin,NDIM)
    DoHisto : Optional[bool]
        Project on the weights on a grid of dimention power(Nbin,NDIM) 
    rangeBin : Optional[array_like[NDIM,2,]]
        The range of each dimentions. 
    
    Returns
    -------
    list_bins :
        list of the bins used for each axis.
    ND_pdf : 
        The power(Nbin,NDIM) cube of the pdf
    TwoD_proba :
        All the 2D pdf.
    OneD_proba :
        All the 1D pdf.
    axe_marg_2D.astype(int) :
        The order of axis used in the 2D pdf.
    axe_marg_1D.astype(int) :
        The order of axis used in the 1D pdf. 
    
    Raises
    ------
    None
    """

    DIM = len(list_points)

    ### Create bins of each dim
    list_bins = np.zeros( [DIM, Nbin] )
    for i, pts in enumerate(list_points):
        try:
            if rangeBin==None:
                list_bins[i] = np.linspace( pts.min(), pts.max(), num=Nbin )
        except ValueError:
            list_bins[i] = np.linspace( rangeBin[0,i], rangeBin[1,i], num=Nbin )

    if DoInterp:
        ### FOR GRID ESTIMATIONS
        ### ND mesh 
        YYY = np.mgrid[ [slice(m, M, Nbin*1j) for m,M in zip( list_bins.min(axis=1), list_bins.max(axis=1) ) ] ]

        ### ND parametter interpolate space 
        ND_proba = griddata( list_points.T, weights, 
                             YYY.T, 
                             method='linear', fill_value=0., rescale=True ) ### nearest ### linear
    elif DoHisto:
        ### FOR MCMC CHAINS
        #print("IN DO HISTOGRAM")
        list_bins_tmp = np.zeros( [DIM, Nbin+1] )
        for i, pts in enumerate(list_points):
            try:
                if rangeBin==None:
                    list_bins_tmp[i] = np.linspace( pts.min(), pts.max(), num=Nbin+1 )
            except ValueError:
                list_bins_tmp[i] = np.linspace( rangeBin[0,i], rangeBin[1,i], num=Nbin+1 )
        ND_proba, _ = np.histogramdd( list_points.T, weights=weights, bins=list_bins_tmp )
        ND_proba = ND_proba.T
    else:
        ### WHEN THE PROBA IS ALREADY COMPUTE
        #ND_proba = weights.reshape( [Nbin, Nbin, Nbin, Nbin, Nbin] )
        ND_proba = weights.reshape( np.ones( DIM, dtype=int )*Nbin )
    
    ND_proba[ np.isnan( ND_proba ) ] = 0. ### just in case
    print( 'NORMALIZATION FACTOR :',ND_proba.sum() )
    ND_pdf = ND_proba.T / ND_proba.sum() ### normalisation 

    ### 2D planes : marginalised proba = n(n-1)/2
    TwoD_proba = np.zeros( [DIM*(DIM-1)//2, Nbin, Nbin ] )
    axe_marg_2D = np.zeros( [DIM*(DIM-1)//2, 2] )
    t = pyDOE.fullfact( np.arange(3,DIM+1) )
    if DIM>3:
        select = np.where( np.product( np.array([ (t[:,d]<t[:,d+1]) for d in range(t.shape[1]-1) ]), axis=0 ) )[0] 
        tselect = t[select]
    else:
        tselect = t
    for i, ax in enumerate( tselect ):
        TwoD_proba[i] = np.squeeze( np.apply_over_axes(np.sum, ND_pdf, ax.astype( int )) )
        axe_marg_2D[i] = np.setdiff1d( range(DIM), ax )

    ### 1D : marginalised proba = n
    OneD_proba = np.zeros( [DIM, Nbin ] )
    axe_marg_1D = np.zeros( DIM )
    t = pyDOE.fullfact( np.arange(2,DIM+1) )
    if DIM>2:
        select = np.where( np.product( np.array([ (t[:,d]<t[:,d+1]) for d in range(t.shape[1]-1) ]), axis=0 ) )[0] 
        tselect
    else:
        tselect=t
    for i, ax in enumerate( tselect ):
        OneD_proba[i] = np.squeeze( np.apply_over_axes(np.sum, ND_pdf, ax.astype( int )) )
        
        axe_marg_1D[i] = np.setdiff1d( range(DIM), ax )
        
    return list_bins, ND_pdf, TwoD_proba, OneD_proba, axe_marg_2D.astype(int), axe_marg_1D.astype(int)


##############################################################################

def find_zeros( x, y ):
    """
    Return the zeros (x_zeros) of f(x)=y
    f(x_zeros)=0
    
    Note
    ----

    Parameters
    ----------
    x : array_like[nSamples,]
        
    y : array_like[nSamples,]
        f(x)=y
    
    Returns
    -------
    zeros : array_like[Nzeros,]
        The list of all the zeros.
    """
    zeros = []
    N_zeros = 0
    N_zeros += ((y)==0).sum()
        
    for ze in range( N_zeros ):
        zeros.append( x[np.where(y==0)[0][ze]] )

    N_zeros += ( np.abs( np.diff( np.sign( y ) ) )==2 ).sum()
    list_other_zeros = np.where( np.abs( np.diff( np.sign( y ) ) )==2 )[0]

    for ze in list_other_zeros:
        ordered = np.argsort( y[ ze:ze+2 ])
        zeros.append( np.interp( 0, y[ ze:ze+2 ][ordered], x[ ze:ze+2 ][ordered] ) )

    zeros = np.sort( zeros )
    return zeros

##############################################################################
def latex(string):
    """
    to make a proper math in plots
    """
    return r'$\rm{' +string+ '}$'
##############################################################################

def plot_triangular( list_points, weights, Nbin=200, param_label='', rangeBin=None, units='', scales='', DoInterp=True, addPoints=True, factor=1.5, DoHisto=False, justContour=False, addHDP=True, title1D=True, numberSigmaContour=3, figAndAxes=None, color=None, addLengend=None ):
    """
    This is an hack of coner.py
    https://github.com/dfm/corner.py/blob/master/docs/index.rst
    I change/adapt to my personal preference and some need. 
    
    It perform a triangular plot for dimentions > 1
    - It can project an MCMC on a grid
    - or interpolate weighted point on a grid
    - lots of option added by hand from the needs
    
    Parameters
    ----------
    list_points : array_like[nPoints,NDIM,]
        Contain the sample of points.
    weights : array_like[nPoints,]
        The weights of each points.
    Nbin : Optional[int]
        Dimention of the final grid.
        
    DoInterp : Optional[bool]
        Interpolate the weights on a grid of dimention power(Nbin,NDIM)
    DoHisto : Optional[bool]
        Project on the weights on a grid of dimention power(Nbin,NDIM) 
        
    param_label : Optional[[NDIM,str]]
        The axis labels. It use the function my_utils.latex
    rangeBin : Optional[array_like[NDIM,2,]]
        The axis range, by default takes the min/max
    units : Optional[[NDIM,str]]
        The str of unit of the axis
    scales : Optional[[NDIM,str]] 'lin' or 'log'
        The scale of each axis
        
    justContour : Optional[bool],
        Just do contour plot, remove the backgroud histogram.
    addHDP : Optional[bool] (typo here)
        Add on the 1D plots the HPD bars and area.
    title1D : Optional[bool] 
        Add the max and =- HPD on the title of each 1D plot
        
    numberSigmaContour : Optional[int] 0, 1, 2, 3
        The number of sigmas contour wanted,
    color : Optional[[NDIM,str]]
        List of colors for the sigmas contours.
    addLengend : Optional[str]
        To name the curent triangular plot. 
    
    figAndAxes : [fig,ax] object
        A given figure and ax objects from a previous plot_triangular to over-plot it.
    
    Returns
    -------
    fig, axes :  [fig,ax] object
        The figure and ax objects from the current plot_triangular to over-plot it later. 
    
    Raises
    ------
    None
    """
    
    if color is None:
        color= 'k'
    
    ### number of dimension
    DIM = len(list_points)
    
    if scales=='':
        scales = [ '' for i in range(DIM) ]
    if units=='':
        units = [ '' for i in range(DIM) ]
    if param_label=='':
        param_label = [ '' for i in range(DIM) ]

    ### compute the stat
    list_bins, ND_pdf, TwoD_proba, OneD_proba, axe_marg_2D, axe_marg_1D = compute_stat( list_points, weights, Nbin=Nbin, rangeBin=rangeBin, DoInterp=DoInterp, DoHisto=DoHisto )
    
    ### PLOT NICE DIMENTIONS
    K = DIM
    #factor = 1.5
    lbdim = 0.5*factor
    trdim = 0.2*factor
    whspace = 0.05
    plotdim = factor*K + factor*(K-1) * whspace
    dim = lbdim + plotdim + trdim
    
    plt.rcParams[ 'font.size' ] = 6*factor
    #plt.rcParams['image.cmap'] =   'Greys' #'plasma' #viridis' #'Greys' #'nipy_spectral'

    if figAndAxes is None:
        fig, axes = plt.subplots( K, K, figsize=(dim,dim) )
    else:
        fig, axes = figAndAxes

    lb = lbdim / dim
    tr = ( lbdim + plotdim ) / dim
    fig.subplots_adjust( left=lb,bottom=lb, right=tr, top=tr, wspace=whspace, hspace=whspace )
    
    COUNT_2D_PANEL = 0

    ### DO EACH PANEL
    for i in range(DIM):
        for j in range(DIM):
            

            ### the panel above the diagonal are empty
            if i < j:
                ax = axes[i,j]
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])

            ### the diagonal panel: 1D
            if i==j:
                ax = axes[i,j]

                ID_axe = np.where( i==axe_marg_1D )[0][0]
                binX = np.squeeze( list_bins[axe_marg_1D[ID_axe]] )
                proba = np.squeeze( OneD_proba[ ID_axe, : ] )
                label = param_label[ axe_marg_1D[ID_axe] ]
                unit = units[ axe_marg_1D[ID_axe] ]
                scale = scales[ axe_marg_1D[ID_axe] ]
                                
                ### the 1D plot
                ax.plot( binX, proba/proba.sum(), lw=1, color=color, label=addLengend )

                ### x limit
                ax.set_xlim( binX.min(), binX.max() )
                ### x labels and ticks
                if i<DIM-1:
                    ax.set_xticklabels([])
                    ax.xaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) )
                else:
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                    ax.xaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) )
                    ax.set_xlabel( label+' '+unit )
                ### y labels and ticks
                ax.set_yticklabels([])
                ax.yaxis.set_major_locator( NullLocator() )

                #q_50,q_84,_,_,q_16,_,_ = sigma123_1D( proba, binX )
                #q_m, q_p = q_50-q_16, q_84-q_50
                #ax.axvline( q_50, color='k', ls="dashed" )
                #ax.axvline( q_84, color='k', ls="dashed" )
                #ax.axvline( q_16, color='k', ls="dashed" )
                
                #avg = np.average( binX, weights=proba )
                #max_proba = binX[ np.where( proba==proba.max() ) ]
                #s0 ,   _ = find_123sigma( binX, proba, 0.   )
                #s1m, s1p = find_123sigma( binX, proba, 0.68 )
                #s2m, s2p = find_123sigma( binX, proba, 0.95 )
                #s3m, s3p = find_123sigma( binX, proba, 0.99 )
                #ax.axvline( avg, color='k', ls="dashed", lw=1 )
                #ax.axvline( max_proba, color='y', ls="dashed", lw=1 )
                #ax.axvline( s0 , color='k', ls="dashed", lw=1 )
                #ax.axvline( s1m, color='r', ls="dashed" )
                #ax.axvline( s2m, color='g', ls="dashed" )
                #ax.axvline( s3m, color='b', ls="dashed" )
                #ax.axvline( s1p, color='r', ls="dashed" )
                #ax.axvline( s2p, color='g', ls="dashed" )
                #ax.axvline( s3p, color='b', ls="dashed" )
                
                ### max
                smax = binX[proba==proba.max()][0]
                p1 = find_proba_limit( proba/proba.sum(), confidence_level=0.68 )
                zeros = find_zeros( binX, proba/proba.sum()-p1 )
                new_binX = np.linspace( binX.min(), binX.max(), 1000 ) 
                new_pdf = np.interp( new_binX, binX, proba/proba.sum() )
                if addHDP:
                    
                    ax.axvline( smax, color='k', ls="dashed", lw=1 ) 
                    ### mean
                    #s0 = (binX*proba).sum()/proba.sum()
                    #ax.axvline( s0, color='b', ls="dashed", lw=1 ) 

                    #s1, p1 = get_HDP( binX, proba/proba.sum() )[0]
                    
                    #print(p1)
                    ax.axhline( p1, color='r', ls="dashed", lw=1 ) 

                    #for ze in zeros:
                    #    ax.axvline( ze, color='r', ls="dashed" ) 
                    ax.axvline( zeros.min(), color='r', ls="dashed", lw=1 ) 
                    ax.axvline( zeros.max(), color='r', ls="dashed", lw=1 ) 

                    ax.fill_between( new_binX, new_pdf*(new_pdf>=p1) )                
                
                
                #s1m, s1p = s1
                ### temporary sampling the proba to compute HDT with pymc3
                #tempo = np.interp( np.random.rand(100000), proba.cumsum()/proba.sum(), binX )
                #s1m, _ = pm.stats.hpd( tempo, alpha=0.32 )
                #tempo = np.interp( np.random.rand(100000), 
                #                   np.concatenate( ([0], proba.cumsum()/proba.sum() ) ), 
                #                   np.concatenate( (binX, [binX[-1]] ) ) )
                #_, s1p = pm.stats.hpd( tempo, alpha=0.32 )
                #ax.axvline( s1m, color='r', ls="dashed" ) 
                #ax.axvline( s1p, color='r', ls="dashed" ) 
                #print( s1m, s1p )
                
                if title1D:
                    # Format the quantile display.
                    fmt = "{{0:{0}}}".format(".2f").format
                    title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
                    #print( smax, s1m, s1p )
                    #print( zeros.min(), smax, zeros.max() )
                    #title = title.format( fmt(q_50), fmt(q_m), fmt(q_p) )
                    #title = title.format( fmt(smax), fmt(np.abs(smax-s1m)), fmt(np.abs(s1p-smax)) )
                    title = title.format( fmt(smax), fmt(np.abs(smax-zeros.min())), fmt(np.abs(smax-zeros.max())) )

                    # Add in the column name if it's given.
                    if scale=='lin':
                        title = "{0} = {1}".format( label, title)
                    else:
                        title = "{0} = {1}".format( r'$\rm{log_{10}(}$'+label+')', title)
                    ax.set_title( title )
                 
                if i==0 and (addLengend is not None):
                    ax.legend( bbox_to_anchor=(1.05, 1) )

            ### 2D : the bellow panels
            if i>j:
                ax = axes[i,j]
                
                ID_axe = np.where( np.prod(axe_marg_2D==[j,i], axis=1) )[0][0]
                binX = np.squeeze( list_bins[ axe_marg_2D[ID_axe][0] ] )
                binY = np.squeeze( list_bins[ axe_marg_2D[ID_axe][1] ] )
                proba = np.squeeze( TwoD_proba[ ID_axe ] ).T
                labelX = param_label[ axe_marg_2D[ID_axe][0] ]
                labelY = param_label[ axe_marg_2D[ID_axe][1] ]
                unitX = units[ axe_marg_2D[ID_axe][0] ]
                unitY = units[ axe_marg_2D[ID_axe][1] ]
                
                if addPoints:
                    ax.plot( list_points[ axe_marg_2D[ID_axe][0] ], list_points[ axe_marg_2D[ID_axe][1] ], 
                             'k.', markerfacecolor='none', markeredgecolor='k', markersize=5, markeredgewidth=0.5 )
                
                if not(justContour):
                    ax.imshow( (proba/proba.max()), origin='lower', 
                               extent=[ binX.min(), binX.max(), 
                                        binY.min(), binY.max()], #)
                               cmap=cm.viridis, 
                               interpolation='gaussian', ) ### nearest
                
                ONE_sigma = so.brentq( find_confidence_interval, 0., 1., args=(proba,0.68) ) 
                TWO_sigma = so.brentq( find_confidence_interval, 0., 1., args=(proba,0.95) ) 
                THREE_sigma = so.brentq( find_confidence_interval, 0., 1., args=(proba,0.99) ) 
                #print('ONE_sigma : ', ONE_sigma)
                X, Y = np.meshgrid( binX, binY )
                #ax.contour( X, Y, proba, 3, levels=[THREE_sigma, TWO_sigma, ONE_sigma], colors=('k', 'grey', 'w') )
                if numberSigmaContour==1:
                    ax.contour( X, Y, proba, 1, levels=[ONE_sigma], colors=(color), zorder=200 )
                elif numberSigmaContour==2:
                    ax.contour( X, Y, proba, 2, levels=[TWO_sigma, ONE_sigma], colors=('g', 'r'), zorder=200 )
                elif numberSigmaContour==3:
                    ax.contour( X, Y, proba, 3, levels=[THREE_sigma, TWO_sigma, ONE_sigma], colors=('b', 'g', 'r'), zorder=200 )

                ###
                x0,x1 = ax.get_xlim()
                y0,y1 = ax.get_ylim()
                ax.set_aspect( (x1-x0)/(y1-y0) )

                if( i<(DIM-1) ):
                    ax.set_xticklabels([])
                    ax.xaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) ) 
                else:
                    ax.xaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) ) 
                    ax.set_xlabel( labelX+' '+unitX )
                    [l.set_rotation(45) for l in ax.get_xticklabels()]
                    
                if(j==0):
                    ax.yaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) )
                    ax.yaxis.set_major_formatter( ScalarFormatter(useMathText=True) )
                    ax.set_ylabel( labelY+' '+unitY )
                    ax.yaxis.set_label_coords( -0.3, 0.5 )
                    ax.yaxis.tick_left()
                    [l.set_rotation(45) for l in ax.get_yticklabels()]
                else:
                    ax.set_yticklabels([])
                    ax.yaxis.set_major_locator( MaxNLocator( 6, prune='lower' ) ) 
    return fig, axes