# Mandelbrot-GPU
An optimized GPU and CPU based Mandelbrot Set image generator. 

This python scirpt will generate a mandelbrot or julia fractal image with a given set of paramters. 

#### Check out the example man(delbrot) and julia pictures.

### Requires:

* numpy
* theano
* sympy
* numexpr
* matplotlib

In order to test it out, run:

    python example.py
  
In order to make your own images

    from mandelbrot import *
    
    limits = create_interval(center=[-0.5,0], radius=1.5)

    mandel_single(max_iter, res, limits, name)
    julia_single(c, func, max_iter, res, 
                 limits, raw , name)
                 
* max_iter: Maximum amount of iterations to perform.
* res: Resolution of image in pixels.
* limits: Area of the Julia/Mandelbrot set to render.
* name: Name of file to create.

##### Julia only

* c: Constant complex point to create Julia set around.
* func: Function to create Julia set for.


##### For example usage of these functions, check out example.py.

