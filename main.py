from models import CNN, save_model, load_model
from preprocessing import preprocess


X0, y0, Y0, X1, y1, Y1 = preprocess()
model = CNN(X0, Y0, X1, Y1, fit=True)

save_model(model)