import yaml
from box import Box

from src.datamodule import CCDataModule
from src.module import CCModule

with open('config.yaml', 'r') as f:
    cfg = Box(yaml.safe_load(f))

print('\nInstantiate datamodule')
datamodule = CCDataModule(
    encoder_name=cfg.encoder_name,
    image_path=cfg.image_path,
    gt_path=cfg.gt_path,
    val_size=cfg.val_size,
    test_size=cfg.test_size,
    batch_size=cfg.batch_size,
    seed=cfg.seed,
)

print('\nLoad model from checkpoint')
ckp_filepath = 'checkpoints/20231221-231007/epoch=2.ckpt'
model = CCModule.load_from_checkpoint(ckp_filepath)

print('\nEvaluating...')
model.eval()
model.to('cuda:0')
model.evaluate(datamodule, 'cuda:0')
print('Done!')