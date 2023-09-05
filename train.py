from pytorch_lightning.cli import LightningCLI

from dataset import WrapperDataset
from model import ClassificationModel
if __name__ == "__main__":

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.batch_size", apply_on="parse")
            parser.link_arguments("data.logits_size", "model.logits_size", apply_on="instantiate")
            # parser.link_arguments("ckpt_path", "model.init_args.ckpt_path")
            parser.link_arguments("data.diagnostic_labels", "model.labels", apply_on="instantiate")
            parser.link_arguments("data.class_weights", "model.class_weights", apply_on="instantiate")

    cli = MyLightningCLI(model_class=ClassificationModel, datamodule_class=WrapperDataset)