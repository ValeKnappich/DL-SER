from data import SERDatamodule
from model import SERClassifier
import pytorch_lightning as pl

dm = SERDatamodule("train.json")
model = SERClassifier()
trainer = pl.Trainer(
    max_epochs=3,
    gpus=1
)
trainer.fit(model, dm)
trainer.save_checkpoint("model.ckpt")