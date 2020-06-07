from py3grads import Grads,GrADSError
from collections import OrderedDict
from itertools import product
import numpy as np
import xarray as xr

class Gradsx(Grads):
    def __call__(self, lines):
        """
        Allow commands to be passed to the GrADS object
        """
        rc_all=[]
        outlines_all=[]
        for gacmd in lines.split("\n"):
            if gacmd=="": continue 
            outlines, rc = self.cmd(gacmd)
            if rc > 0:
                print('\n'.join(outlines))
                raise GrADSError('GrADS returned rc='+str(rc)
                                 +' for the following command:\n'+gacmd)
            rc_all.append(rc)
            outlines_all.append(outlines)
        return outlines_all, rc_all
    
    def expx(self, expr,chunk=False):
        """
        Export a GrADS field to a Numpy array. Since only up to 2-dimensional
        data can be written out by GrADS, requesting arrays of rank > 2 will be
        less efficient than defining the same array in GrADS.
        Args:
            expr: GrADS expression representing the field to be exported.
        """
        # Get the current environment
        env = self.env()
        dimnames = ('x','y','z','t','e') # ordered by GrADS read efficiency
        # Detect which dimensions are varying
        dims = [dim for dim in dimnames if not getattr(env, dim+'fixed')]
        # We can only display/output data from GrADS up to 2 dimensions at a
        # time, so for rank > 2, we must fix the extra dimensions. For best
        # efficiency, always select the two fastest dimensions to vary.
        varying, fixed = dims[:2], dims[2:]
        # Varying dimensions must be ordered identically to GrADS fwrite output
        fwrite_order = ['z','y','x','t','e']
        varying.sort(key=lambda dim: fwrite_order.index(dim))
        output_dims = varying + fixed
        # For common cases, it is desirable to enforce a certain dimension
        # order in the output array for the first two axes
        output_orders2D = OrderedDict([
            ('xy', ['y','x']), ('xz', ['z','x']), ('yz', ['z','y']),
            ('xt', ['t','x']), ('yt', ['y','t']), ('zt', ['z','t'])
        ])
        # Check for 2D base dimensions in order of preference
        for first2, order in output_orders2D.items():
            if set(first2).issubset(dims):
                ordered_dims = order + [d for d in dims if d not in order]
                break
        else:
            ordered_dims = dims
        # Read data into Numpy array
        if len(dims) <= 2:
            arr = self._read_array(expr, varying)
        else:
            dimvals = {}
            for dim in dims:
                mn, mx = getattr(env, dim+'i')
                dimvals[dim] = range(mn, mx+1)
            # Sets of fixed coordinates for which to request arrays while the
            # first two (most efficient) dimensions vary
            coordinates = product(*[dimvals[dim] for dim in fixed])
            arr = None # Need to wait to define until we know shape of arr1D
            for coords in coordinates:
                # Set fixed dimemsions and get array indices
                idx = []
                for dim, c in zip(fixed, coords):
                    self.cmd('set {dim} {c}'.format(dim=dim, c=c))
                    idx.append(dimvals[dim].index(c))
                # Get 2D array
                arr2D = self._read_array(expr, varying)
                # Define full data array
                if arr is None:
                    arr = np.zeros(arr2D.shape + tuple(len(dimvals[d]) for d in fixed))
                # Assign data along first two dimensions
                arr[(slice(None), slice(None)) + tuple(idx)] = arr2D
        # Re-order axes if necessary
        axes = [(i, output_dims.index(d)) for i, d in zip(range(len(dims)), ordered_dims)]
        swapped = []
        for a1, a2 in axes:
            pair = sorted([a1, a2])
            if a1 != a2 and pair not in swapped:
                arr = np.swapaxes(arr, a1, a2)
                swapped.append(pair)
                
        # to xarray
        xarr=self._to_xarray(expr,arr,env,ordered_dims,chunk)
 
        # Restore original GrADS dimension environment
        for dim in dims:
            mn, mx = getattr(env, dim)
            self.cmd('set {dim} {mn} {mx}'.format(dim=dim, mn=mn, mx=mx))
        return xarr
    
    def _to_xarray(self,expr,arr,env,dims,chunk=False):
        """"
        to xarray
        """
        geodims={"x":"lon","y":"lat","z":"lev","t":"time","e":"ens"}
        axisname=[]
        axisval={}
        for dim in dims:
            gdim=geodims[dim]
            ax=self._get_axis(env,dim,gdim)
            axisname.append(gdim)
            axisval[gdim]=ax
        if chunk:
        	xarr=xr.DataArray(arr,name=expr,coords=axisval,dims=axisname).chunk()
        else:
        	xarr=xr.DataArray(arr,name=expr,coords=axisval,dims=axisname)	
        return xarr

    def _get_axis(self,env,dim,gdim):
        """
        env : envrionment 
        dim, z,y,x,t,e
        gdims lev,lat,lon,time, ens
        """
        if dim=="t":
            mn, mx = getattr(env, "ti")
            ax=[]
            for m in np.arange(mn,mx+1):
                self.cmd('set t {m}'.format(m=m))
                envtemp=self.env()
                ax.append(getattr(envtemp,"time"))
        else:
            for d in ['z','y','x','t','e']:
                if d == dim:
                    mn, mx = getattr(env, dim)
                    self.cmd('set {d} {mn} {mx}'.format(d=d, mn=mn, mx=mx))
                else:
                    self.cmd('set {d} 1'.format(d=d))

            ax=self.exp(gdim)

        return ax
                
                
