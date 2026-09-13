"""Exact enumeration of alpha-compositing expectation; no renderer assumptions."""
import itertools
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'methods'))
from opacity_expectation import OpacityExpectation

def composite(alpha, colour, background):
    trans=1.;value=np.zeros(3)
    for a,c in zip(alpha,colour):
        value+=trans*a*c;trans*=1-a
    return value+trans*background

alpha=np.array([.2,.7,.4]);q=np.array([.6,.8,.9])
colour=np.array([[.1,.4,.8],[.8,.2,.5],[.6,.6,.1]])
background=np.array([.4,.2,.1]);expected=np.zeros(3)
for mask in itertools.product([0,1],repeat=3):
    m=np.array(mask);weight=np.prod(np.where(m,q,1-q))
    expected+=weight*composite(m*alpha,colour,background)
assert np.allclose(expected,composite(q*alpha,colour,background),atol=1e-12)
class Example:
    get_opacity=alpha
    sentinel=42
g=Example();proxy=OpacityExpectation(g)
assert proxy.sentinel==42 and np.allclose(proxy.get_opacity,.85*alpha)
assert np.array_equal(g.get_opacity,alpha)
assert np.array_equal(OpacityExpectation(g,1).get_opacity,alpha)
print('Exact compositing expectation, proxy identity and non-mutation checks passed.')
