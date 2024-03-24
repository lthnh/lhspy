import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

app = pg.mkQApp('Test')
pg.setConfigOptions(antialias=True)
p = pg.plot()
curve = p.plot()
data = np.random.normal(size=(10,1000))
ptr = 0
def update():
    global curve, data, ptr
    curve.setData(data[ptr%10])
    if ptr == 0:
        p.enableAutoRange('xy', False)  ## stop auto-scaling after the first data set is plotted
    ptr += 1
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(50)
p.show()

app.exec()