from pytorch_lightning.cli import LightningCLI

from dataset import WrapperDataset

if __name__ == "__main__":

    class MyLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.batch_size", "model.init_args.batch_size", apply_on="instantiate")
            parser.link_arguments("data.logits_size", "model.init_args.logits_size", apply_on="instantiate")
            parser.link_arguments("trainer.devices", "model.init_args.world_size", apply_on="parse")
            parser.link_arguments("ckpt_path", "model.init_args.ckpt_path", apply_on="parse")
            parser.link_arguments("data.diagnostic_labels", "model.init_args.labels", apply_on="instantiate")
            parser.link_arguments("data.class_weights", "model.init_args.class_weights", apply_on="instantiate")

    cli = MyLightningCLI(datamodule_class=WrapperDataset)