# author: ddthuan@pdu.edu.vn

from utils.basic_lib import *
import config as cf
from dataset import train_data_loader
from engine import train, save_model

num_epochs = cf.EPOCHS

for epoch in range(num_epochs):
    start = time.time()
    train_loss = train(train_data_loader)
    print(f"Epoch #{epoch} loss: {train_loss}")
    end = time.time()
    print(f"Took {(end - start) / 60} minutes for epoch {epoch}")

save_model()



