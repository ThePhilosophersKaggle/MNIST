from models import CNN
from preprocessing import preprocess


X0, y0, Y0, X1, y1, Y1 = preprocess()
CNN(X0, Y0, X1, Y1, fit=True)
