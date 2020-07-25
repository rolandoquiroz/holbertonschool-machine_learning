#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly))

poly = [5, 3, '0', 1]
print(poly_integral(poly))

poly = ['5', 3, 1]
print(poly_integral(poly))

poly = ['5', 3, 1]
C = 5.5
print(poly_integral(poly, C))

poly = [0]
C = 5
print(poly_integral(poly, C))

poly = [69]
C = 5
print(poly_integral(poly, C))

poly = [0, 6]
C = 666
print(poly_integral(poly, C))

poly = [0, 0, 1]
C = 13
print(poly_integral(poly, C))
