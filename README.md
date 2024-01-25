# py3gradsx
Extension of [py3grads](https://github.com/meridionaljet/py3grads) to export xarray data array 


## DEPENDENCIES

py3gradsx requires py3grads and xarray
py3grads requires NumPy and a working GrADS installation.

## USAGE
```
from py3gradsx import Gradsx
ga = Gradsx(verbose=False)
```

py3gradsx can do what py3grads can do.

There are two extensions.

(1) pygrads allows the following multiline commands

```
 script = """
           open mslp.ctl
           set lat 20 50
           set lon 120 140
           define mslpm=ave(mslp,time=1jan2020,time=31jan2020)
           """
ga(script)
```

(2) export to xarray data array

```
mslp = ga.expx("mslpm")
```


