
#import main.assignment_1 as assignment_1
from ..main import assignment4
import numpy as np

A = np.array([[10, -1, 2, 0],
              [-1, 11, -1, 3],
              [2, -1, 10, -1],
              [0, 3, -1, 8]])
b = np.array([6.0, 25.0, -11.0, 15.0])


T,c = assignment4.linear_system_to_jacobi_form(A,b)

initial = np.zeros_like(b)

sol1,iterations = assignment4.Jacobi(T,c,initial ,1000,1e-3)
print("Solution 1 in "+str(iterations)+" iterations: ")
i = 1
for row in sol1:
    print("x"+str(i)+": \t"+ str(row))
    i+=1
print("\n")


initial = np.zeros_like(b)

sol2,its2 = assignment4.GaussSeidel(A,b,initial,1000,1e-3)
print("Solution 2 in "+str(its2)+" iterations: ")
i = 1
for row in sol2:
    print("x"+str(i)+": \t"+ str(row))
    i+=1
print("\n")

initial = np.zeros_like(b)
omega = 1.2

sol3,its3 = assignment4.SOR(A,b,initial,omega,1000,1e-3)
print("Solution 3 in "+str(its3)+" iterations: ")
i = 1
for row in sol3:
    print("x"+str(i)+": \t"+ str(row))
    i+=1
print("\n")


A = np.array([[3.3330 ,15920,-10.333],
             [2.2220, 16.710, 9.6120],
             [1.5611, 5.1791, 1.6852]
             ])
b = np.array([15913,28.544,8.4254])


t = 5
sol4,cond = assignment4.iterative_refinement(A,b,1000,1e-3,t)
print("Solution 4: "+str(sol4))
print("Condition: "+str(cond))