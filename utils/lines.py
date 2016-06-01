# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 19:41:51 2015

@author: caoym
"""
import numpy

def line_intersect(p0, p1, m0=None, m1=None, q0=None, q1=None):
    ''' intersect 2 lines given 2 points and (either associated slopes or one extra point)
    Inputs:
        p0 - first point of first line [x,y]
        p1 - fist point of second line [x,y]
        m0 - slope of first line
        m1 - slope of second line
        q0 - second point of first line [x,y]
        q1 - second point of second line [x,y]
    '''
    if m0 is  None:
        if q0 is None:
            raise ValueError('either m0 or q0 is needed')
        dy = q0[1] - p0[1]
        dx = q0[0] - p0[0]
        lhs0 = [-dy, dx]
        rhs0 = p0[1] * dx - dy * p0[0]
    else:
        lhs0 = [-m0, 1]
        rhs0 = p0[1] - m0 * p0[0]

    if m1 is  None:
        if q1 is None:
            raise ValueError('either m1 or q1 is needed')
        dy = q1[1] - p1[1]
        dx = q1[0] - p1[0]
        lhs1 = [-dy, dx]
        rhs1 = p1[1] * dx - dy * p1[0]
    else:
        lhs1 = [-m1, 1]
        rhs1 = p1[1] - m1 * p1[0]

    a = numpy.array([lhs0, 
                  lhs1])

    b = numpy.array([rhs0, 
                  rhs1])
    try:
        px = numpy.linalg.solve(a, b)
    except:
        px = numpy.array([numpy.nan, numpy.nan])

    return px