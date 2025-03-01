import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Data
revenue = np.array([5, 10, 20, 8, 6, 4, 12, 15])  # Pendapatan dalam ribuan $
sales = np.array([27, 46, 73, 40, 28, 26, 60, 59])  # Penjualan pizza dalam ribuan

slope, intercept, r_value, p_value, std_err = linregress(revenue, sales)

predicted_sales = slope * revenue + intercept

plt.scatter(revenue, sales, color='blue', label='Data aktual')
plt.plot(revenue, predicted_sales, color='red', label='Garis regresi')
plt.title('Scatter Plot Pendapatan vs Penjualan Pizza')
plt.xlabel('Pendapatan Rata-rata (1000$)')
plt.ylabel('Penjualan Pizza (1000 buah)')
plt.legend()
plt.grid(True)
plt.show()

correlation = np.corrcoef(revenue, sales)[0, 1]
print(f"Koefisien Korelasi (R): {correlation:.2f}")

ssr = np.sum((predicted_sales - np.mean(sales))**2)
sse = np.sum((sales - predicted_sales)**2)
sst = np.sum((sales - np.mean(sales))**2)
r_squared = r_value**2

print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")
print(f"SSR: {ssr:.2f}")
print(f"SSE: {sse:.2f}")
print(f"SST: {sst:.2f}")
print(f"R-squared: {r_squared:.2f}")
