from mandelbrot import *

# Create an interval around a point with a given radius
limits = create_interval(center=[-0.5,0], radius=1.5)

julia_single(c=(-0, 0.0), 
             func='z**2 + c',
             max_iter=100, res=1000, limits=limits, raw=False, name='julia-test')


mandel_single(max_iter=100, res=1000, limits=limits, name='mandelbrot-test')