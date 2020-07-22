#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

person = np.array(['Farrah', 'Fred', 'Felicia'])
my_color = np.array(['red', 'yellow', '#ff8000', '#ffe5b4'])
my_label = np.array(['apples', 'bananas', 'oranges', 'peaches'])
my_bottom = 0

for i in range(4):
    if i > 0:
        my_bottom += fruit[i - 1]
    plt.bar(person, fruit[i], color=my_color[i], bottom=my_bottom,
            label=my_label[i], width=0.5)

plt.ylim(0, 80)
plt.yticks(np.arange(0, 90, 10))
plt.legend(loc='upper right')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.show()
