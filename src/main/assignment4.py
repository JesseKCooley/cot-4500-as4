import numpy as np


"""
Input an np Array A representing a system of linear equations and b an np array representing the solution vector
"""
def linear_system_to_jacobi_form(A, b):
    n = len(b)
    T = np.zeros((n, n))
    c = np.zeros(n)
    
    for i in range(n):
        T[i, i] = 0  # Diagonal elements of T are always 0 in Jacobi iteration
        for j in range(n):
            if j != i:
                T[i, j] = -A[i, j] / A[i, i]
        c[i] = b[i] / A[i, i]
    
    return T, c


"""
#Input:
# T - an np array representing a linear system of equations (in jacobi form!)
# c - an np array representing the constant vector b
# initial - an np array of the initial guess
# max iterations, desired tolerance
"""
def Jacobi(T,c, initial, max_iterations,tolerance):
   
    x = initial.copy()
    
    for k in range(max_iterations):
        x_new = np.dot(T, x) + c
        
        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < tolerance:
            return x_new,k
        
        x = x_new

    print("Maximum iterations reached without convergence")
    return x,max_iterations


"""
#Input:
# T - an np array representing a linear system of equations
# c - an np array representing the constant vector b
# initial - an np array of the initial guess
# max iterations, desired tolerance
"""
def GaussSeidel(A,b, initial_guess, max_iterations,tolerance):
    n = len(b)
    x = initial_guess.copy()
    xo = initial_guess.copy()
    
    for k in range(1, max_iterations + 1):
        for i in range(n):
            summation = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], xo[i+1:])
            x[i] = (b[i] - summation) / A[i, i]
        
        if np.linalg.norm(x - xo) < tolerance:
            return x,k
        
        xo = x.copy()
    
    print("Max iterations exceeded")
    return xo,max_iterations

"""
Input:
A- an np array representing the system of linear equations
b - the solution vector
omega - relaxation parameter
tolerance, and max iterations
"""
def SOR(A,b, initial_guess, omega,max_iterations,tolerance):
    
    n = len(b)
    x = initial_guess.copy()
    xo = initial_guess.copy()
    
    for k in range(1, max_iterations + 1):
        for i in range(n):
            summation = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], xo[i+1:])
            x[i] = (1 - omega) * xo[i] + (omega / A[i, i]) * (b[i] - summation)
        
        if np.linalg.norm(x - xo) < tolerance:
            return x,k
        
        xo = x.copy()

    print("Max iterations exceeded")
    return xo,max_iterations

"""
Not implied by the HW, but might be useful:
A (ndarray): Coefficient matrix of the linear system.
        b (ndarray): Constant vector of the linear system.
        initial_guess (ndarray): Initial guess for the solution.
        tolerance (float): Tolerance for convergence.
        max_iterations (int): Maximum number of iterations for each omega value.
        omega_range (tuple): Range of omega values to explore. Default is (1, 2).
        num_steps (int): Number of steps to divide the omega range. Default is 100.
"""
def find_optimal_omega(A, b, initial_guess, tolerance, max_iterations, omega_range=(1, 2), num_steps=100):
    omegas = np.linspace(omega_range[0], omega_range[1], num_steps)
    min_iterations = float('inf')
    optimal_omega = None
    
    for omega in omegas:
        _, iterations = SOR(A, b, initial_guess, omega, max_iterations,tolerance)
        if iterations < min_iterations:
            min_iterations = iterations
            optimal_omega = omega
    
    return optimal_omega


"""
   A (ndarray): Coefficient matrix of the linear system.
        b (ndarray): Constant vector of the linear system.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Tolerance for convergence.
        precision (int): Number of digits of precision.
"""

def iterative_refinement(A,b,max_iterations,tolerance,precision):
    n = len(b)
    
    # Step 0: Solve the system Ax = b for x by Gaussian elimination
    x = np.linalg.solve(A, b)
    
    # Step 1: Initialize iteration counter and condition
    k = 1
    condition = None
    
    while k <= max_iterations:
        # Step 3: Calculate r
        r = b - np.dot(A, x)
        
        # Step 4: Solve the linear system Ay = r by Gaussian elimination
        y = np.linalg.solve(A, r)
        
        # Step 5: Update x
        xx = x + y
        
        # Step 6: Calculate condition if k = 1
        if k == 1:
            condition = np.linalg.norm(y, ord=np.inf) / np.linalg.norm(x, ord=np.inf) * 10 ** precision
        
        # Step 7: Check convergence
        if np.linalg.norm(xx - x, ord=np.inf) < tolerance:
            return xx, condition
        
        # Step 8: Update iteration counter
        k += 1
        
        # Step 9: Update x for next iteration
        x = xx.copy()
    
    # Maximum number of iterations exceeded
    print("Maximum number of iterations exceeded")
    return x, condition