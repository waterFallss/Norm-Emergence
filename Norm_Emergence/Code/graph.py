import matplotlib.pyplot as plt

x = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
y = [1,3,7,8,4,9,8,7,6,5]

plt.plot(x, y)
plt.xlabel("Number of Episodes", fontsize=14)
plt.ylabel("Number of Convergence", fontsize=14)
plt.show()