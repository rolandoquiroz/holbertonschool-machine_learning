#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

# plot all 5 previous graphs in one figure
# plt.figure(figsize=(93/16, 70/16))

params = {'axes.titlesize': 'x-small', 'axes.labelsize': 'x-small',
          'legend.fontsize': 'x-small'}
plt.rcParams.update(params)

plt.subplot(321)
plt.plot(y0, color='red', linestyle='-')
plt.xlim(0, 10)
plt.xticks(np.arange(0, 11, 2))
plt.yticks(np.arange(0, 1500, 500))

plt.subplot(322)
plt.scatter(x1, y1, color='magenta', marker='.')
plt.xlabel('Height (in)')
plt.ylabel('Weight (lbs)')
plt.title("Men's Height vs Weight")

plt.subplot(323)
plt.plot(x2, y2)
plt.xlim(0, 28650)
plt.yscale('log')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of C-14')

plt.subplot(324)
plt.axis([0, 20000, 0, 1])
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g-', label='Ra-226')
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
plt.legend(loc='upper right')

plt.subplot(313)
plt.hist(student_grades, 10, (0, 100), edgecolor='black')
plt.axis([0, 100, 0, 30])
plt.xticks(np.arange(0, 110, 10))
plt.yticks(np.arange(0, 40, 10))
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')

plt.suptitle('All in One', x=0.485, y=0.93)
plt.subplots_adjust(wspace=0.355, hspace=1)
plt.show()
