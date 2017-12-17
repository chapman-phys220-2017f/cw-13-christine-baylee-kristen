#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Name: Baylee Mumma, Christine Outlaw
# Email: mumma103@mail.chapman.edu
# Course: PHYS220/MATH220/CPSC220 Fall 2017
# Assignment: CLASSWORK 12

import numpy as np
import matplotlib.pyplot as plt
import numba as nb

def y_dot(t,y,x,nu,F):
    """y_dot(t,y,x,nu,F)
    The differential equation given for the dot derivative of y.
    """
    return -(nu*y)+x-(x**3)+(F*np.cos(t))

def x_dot(y):
    """x_dot(y)
    The differential equation given for the dot derivative of x.
    """
    return y

@nb.jit
def rk4(a, b, x0, y0, nu=0, F=0, xdot = x_dot, ydot = y_dot):
    """rk(a, b, x0, y0, nu=0, F=0, xdot = x_dot, ydot = y_dot)
    
    Args:
        a  (float) : Lower bound, t = a*2*pi
        b  (float) : Upper bound, t = b*2*pi
        x0 (float) : Initial position of ball
        y0 (float) : Initial velocity of ball
        nu (float) : Constant damping coefficient
        F  (float) : Constant force amplitude coefficient
        xdot (function) : Part of the differential equation
        ydot (function) : Part of the differential equation
    
    Returns:
        t (array) : Array over the time interval with equal dt = .001
        x (array) : Array containing the position of the ball at each time in the time array
        y (array) : Array containing the velocity of the ball at each time in the time array
    """
    dt = 0.001
    start = 2*a*np.pi
    end = 2*b*np.pi
    n = int(np.ceil((end-start)/dt))
    t = np.linspace(start,end,n)
    x = np.zeros(n)
    y = np.zeros(n)
    x_dot_vec = np.zeros(n)
    y_dot_vec = np.zeros(n)
    x[0] = x0
    y[0] = y0
    for k in range(n):
        x_dot_vec[k] = x_dot(y[k])
        y_dot_vec[k] = ydot(t[k],y[k],x[k],nu,F)
        if k == n-1:
            break
        else:
            k1y = dt*ydot(t[k],y[k],x[k],nu,F)
            k2y = dt*ydot((t[k]+dt/2),(y[k]+k1y/2),x[k],nu,F)
            k3y = dt*ydot((t[k]+dt/2),(y[k]+k2y/2),x[k],nu,F)
            k4y = dt*ydot((t[k]+dt),(y[k]+k3y),x[k],nu,F)
            rky = (k1y+(2*k2y)+(2*k3y)+k4y)/6
            y[k+1] = y[k]+rky
            k1x = dt*xdot(y[k])
            k2x = dt*xdot(y[k]+k1x/2)
            k3x = dt*xdot(y[k]+k2x/2)
            k4x = dt*xdot(y[k]+k3x)
            rkx = (k1x+(2*k2x)+(2*k3x)+k4x)/6
            x[k+1] = x[k]+rkx
    return (t,x,y)

def plot(t,x,title=""):
    """plot(x,t,title='')
    Generate a plot of a time-valued function.
    
    Args:
        t (array) : Array of time domain
        x (array) : Array of values representing position
        title (string) : Title of the plot

    Returns:
        A plot of a function with respect to time.
        """
    plt.plot(t,x)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title(title, fontsize = 20)
    plt.show()

def parametric(x,y,title = ''):
    """parametric(x,y,title = '')
    Generate a parametric plot of velocity with respect to position.
    
    Args:
        x (array) : Postion of ball with respect to time
        y (array) : Velocity of ball with respect to time
        title (array) : Title of plot

    Returns:
        A parametric plot.
        """
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('$\dot{x}$')
    plt.title(title, fontsize = 20)
    plt.show()

def scatter(x,y,n,title = ''):
    """scatter(x,y,n,title = '')
    Generate a Poincare section of the parametric curve.
    
   Args:
        x (array) : Position with respect to time
        y (array) : Velocity with respect to time
        n (int)   : Number of points in scatter plot
        title (string) : Title of plot
   
   Returns:
       A scatter plot with points at every integer multiple of 2*pi*n for n[0,50].
       """
    ax = plt.subplot
    plt.title(title, fontsize = 20)
    plt.xlabel('x')
    plt.ylabel('$\dot{x}$')
    for k in range(n):
        plt.scatter(x[6*k],y[6*k])
    plt.show()