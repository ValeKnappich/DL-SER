from data import SERDatamodule
from model import SERClassifier


model = SERClassifier.load_from_checkpoint(checkpoint_path="model.ckpt").cuda()
dm = SERDatamodule("dev.json", split=False, train=False)
labelled = model.transform(dm.train_dataloader())
json.dump(labelled, open("dev_labelled.json", "w"))