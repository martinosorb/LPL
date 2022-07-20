import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

for fname in sys.argv[1:]:
    path = Path(fname)
    name = path.stem
    _, acc = np.loadtxt(path, skiprows=1, delimiter=',', unpack=True)
    plt.plot(acc, label=name)

plt.legend()
plt.ylabel("Accuracy of linear decoder")
plt.xlabel("Layer decoded")

plt.show()
plt.savefig("layer_accuracies.png")
