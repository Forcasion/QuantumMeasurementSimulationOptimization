import csv
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True, precision=4)

max_entanglement = 5
max_measurements = 10
count = np.zeros(max_measurements)

records = np.zeros((max_entanglement, max_measurements))

for entanglement in range(max_entanglement):
    filename = f"summary/ascending_ent{entanglement+1}.0_merged.csv"

    total = 0

    with open(filename) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue

            value = row[2].strip()
            ops = int(row[1].strip())
            if value == "inf":
                continue
            total += 1
            records[entanglement][ops-1] += 1

    for i in range(max_measurements):
        records[entanglement][i] = records[entanglement][i]/total


print(records)

rows, cols = records.shape
Y, X = np.meshgrid(np.arange(cols), np.arange(rows))  # X=columns, Y=rows
Z = records

fig = plt.figure(figsize=(9, 6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k', linewidth=0.3, alpha=0.9)

ax.set_xlabel('Log. Negativity')
ax.set_zlabel('Detected Fraction')
ax.set_ylabel('No. Measurements')
ax.set_title('Detection')
fig.colorbar(surf, ax=ax, shrink=0.5)

plt.tight_layout()
plt.show()