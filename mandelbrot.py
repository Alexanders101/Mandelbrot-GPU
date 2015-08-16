from numpy import *
import numpy as np
from time import time
from matplotlib import pyplot
import theano.tensor as T
from theano import function, config, shared
from sympy import expand_complex
from numexpr import evaluate
import numexpr as ne


def manelbrot(max_iter=20, res=1000, limits=[[-1.4, 1.4], [-2, 0.8]]):

    y, x = ogrid[limits[0][0]:limits[0][1]:res*1j,
                 limits[1][0]:limits[1][1]:res*1j]

    c = (x+y*1j).astype(np.complex64)
    shape = c.real.shape
    c_real_t = shared(c.real)
    c_imag_t = shared(c.imag)
    del c
    del x
    del y

    #################################################################

    z_real_t = shared(zeros(shape, dtype=np.float32))
    z_imag_t = shared(zeros(shape, dtype=np.float32))

    w_real = (z_real_t**2 - z_imag_t**2) + c_real_t
    w_imag = (2.0*z_real_t*z_imag_t) + c_imag_t

    mandel = function([], updates=[(z_real_t, w_real), (z_imag_t, w_imag)])

    #################################################################

    ww = ((z_real_t**2) + (z_imag_t**2))
    diverge_t = shared(zeros(shape, dtype=np.float32))

    conjug = function([], updates=[(diverge_t, ww)])

    ###############################################################

    divtime_t = shared(max_iter + zeros(shape, dtype=np.int32))
    i_t = shared(np.int32(0))
    max_iter_t = shared(np.int32(max_iter))
    www = (max_iter_t - i_t) * ((diverge_t > 4) & (divtime_t >= max_iter_t))
    make_image = function([], updates=[(i_t, i_t + 1),
                                       (divtime_t, divtime_t - www)])

    for i in range(max_iter):
        mandel()
        conjug()
        make_image()
        print('iteration:\t%i' % (i+1))
    return divtime_t.get_value()


def julia(c, func, max_iter=20, res=1000, limits=[[-1.4, 1.4], [-2, 0.8]]):
    if type(func) is str:
        func = expand_function(func)
    y, x = ogrid[limits[0][0]:limits[0][1]:res*1j,
                 limits[1][0]:limits[1][1]:res*1j]

    z = (x+y*1j).astype(np.complex64)
    shape = z.real.shape
    z_real_t = shared(z.real)
    z_imag_t = shared(z.imag)
    del z
    del x
    del y

    #################################################################

    c_real_t = shared(c[0] + zeros(shape, dtype=np.float32))
    c_imag_t = shared(c[1] + zeros(shape, dtype=np.float32))

    w_real = eval(func[0])
    w_imag = eval(func[1])

    mandel = function([], updates=[(z_real_t, T.cast(w_real, 'float32')), (z_imag_t, T.cast(w_imag, 'float32'))])

    #################################################################

    ww = ((z_real_t**2) + (z_imag_t**2))
    diverge_t = shared(zeros(shape, dtype=np.float32))

    conjug = function([], updates=[(diverge_t, ww)])

    ###############################################################

    divtime_t = shared(max_iter + zeros(shape, dtype=np.int32))
    i_t = shared(np.int32(0))
    max_iter_t = shared(np.int32(max_iter))
    www = (max_iter_t - i_t) * ((diverge_t > 4) & (divtime_t >= max_iter_t))
    make_image = function([], updates=[(i_t, i_t + 1),
                                       (divtime_t, divtime_t - www)])

    for i in range(max_iter):
        mandel()
        conjug()
        make_image()
        print('iteration:\t%i' % (i+1))
    return divtime_t.get_value()

def julia_raw(c, func, max_iter=20, res=1000, limits=[[-1.4, 1.4], [-2, 0.8]]):
    y, x = ogrid[limits[0][0]:limits[0][1]:res*1j,
                 limits[1][0]:limits[1][1]:res*1j]

    z = (x+y*1j).astype(np.complex64)
    del x
    del y
    shape = z.shape

    c = c * ones(shape, dtype=complex)
    divtime = max_iter + zeros(shape, dtype=int)

    for i in range(max_iter):
        z = evaluate(func)
        diverge = evaluate('z.real**2 + z.imag**2')
        divtime[(diverge > 4) & (divtime==max_iter)] = i
        print('iteration:\t%i' % (i+1))
    return divtime




def get_quadrants(limits):
    y_dif = (limits[0][1] - limits[0][0]) / 2.0
    x_dif = (limits[1][1] - limits[1][0]) / 2.0
    quadrants = [[[ limits[0][0] + y_dif, limits[0][1] ], [ limits[1][0] + x_dif, limits[1][1] ]],
                 [[ limits[0][0] + y_dif, limits[0][1] ], [ limits[1][0], limits[1][1] - x_dif ]],
                 [[ limits[0][0], limits[0][1] - y_dif ], [ limits[1][0], limits[1][1] - x_dif ]],
                 [[ limits[0][0], limits[0][1] - y_dif ], [ limits[1][0] + x_dif, limits[1][1] ]]]
    return quadrants


def mandel_single(max_iter, res, limits):
    t0 = time()
    man = manelbrot(max_iter, res, limits)
    t1 = time()
    print('Done! Took %.2f seconds.' % (t1-t0))
    pyplot.imsave('man.png', man)


def mandel_multi(max_iter, res, limits, base_name='mandel'):
    quadrants = get_quadrants(limits)
    for key, quadrant in enumerate(quadrants):
        t0 = time()
        man = manelbrot(max_iter, res, quadrant)
        t1 = time()
        print('Done with quadrant %i! Took %.2f seconds.' % (key+1, t1-t0))
        pyplot.imsave('%s_%i.png' % (base_name, key+1), man)


def julia_single(c, func, max_iter, res, limits, raw=False):
    t0 = time()
    if raw:
        man = julia_raw(complex(*c), func, max_iter, res, limits)
    else:
        try:
            man = julia(c, func, max_iter, res, limits)
        except:
            print('Couldnt compile function, using raw calculation.')
            man = julia_raw(complex(*c), func, max_iter, res, limits)
    t1 = time()
    print('Done! Took %.2f seconds.' % (t1-t0))
    pyplot.imsave('man.png', man)


def julia_multi(c, func, max_iter, res, limits, raw=False, base_name='julia'):
    quadrants = get_quadrants(limits)
    for key, quadrant in enumerate(quadrants):
        t0 = time()
        if raw:
            man = julia_raw(complex(*c), func, max_iter, res, limits)
        else:
            man = julia(c, func, max_iter, res, limits)
        t1 = time()
        print('Done with quadrant %i! Took %.2f seconds.' % (key+1, t1-t0))
        pyplot.imsave('%s_%i.png' % (base_name, key+1), man)


def expand_function(func):
    func = expand_complex(func).as_real_imag()
    result = []
    for part in func:
        part = str(part)
        part = part.replace('re(c)', 'c_real_t')
        part = part.replace('im(c)', 'c_imag_t')
        part = part.replace('re(z)', 'z_real_t')
        part = part.replace('im(z)', 'z_imag_t')
        part = part.replace('abs(', 'T.abs_(')
        part = part.replace('exp(', 'T.exp(')
        part = part.replace('inv(', 'T.inv(')
        part = part.replace('sqr(', 'T.sqr(')
        part = part.replace('sqrt(', 'T.sqrt(')
        part = part.replace('cos(', 'T.cos(')
        part = part.replace('sin(', 'T.sin(')
        part = part.replace('cosh(', 'T.cosh(')
        part = part.replace('sinh(', 'T.sinh(')
        part = part.replace('log(', 'T.log(')
        part = part.replace('arg(', 'T.angle(')
        part = part.replace('I', '1j')
        part = part.replace('im(', 'T.imag(')
        part = part.replace('re(', 'T.real(')
        result.append(part)
    return result


def create_interval(center, distance):
    x = center[0]
    y = -center[1]

    return [[y - distance, y + distance], [x - distance, x + distance]]
# man = mandel_array(5000, 500, limits)
# limits = ((-1.5, 1.5), (-2.0, 1))
# limits = create_interval([-0.1011, 0.9563], 0.1)
limits = create_interval([0,0], 1.5)

# julia_single((-0, 0.0), 
#               'z**2 + c',
#               100, 1000, limits, raw=False)
mandel_single(100, 1000, limits)

# t0 = time()
# man = mandel_array(100, 1000, limits)
# # man = julia_set(, lambda z, c: z**2 + c, 300, 5000, limits)
# t1 = time()
# print('Done! Took %.2f seconds.' % (t1-t0))
# pyplot.imsave('man.png', man)
# # pyplot.imshow(man)