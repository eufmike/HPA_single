import sys
import argparse
import yaml
from func.trainer import Trainer, Trainer_ln, Train_Project


def dltrain(args):
    config_path = args.config_yaml
    config = yaml.safe_load(open(config_path, "r"))

    if args.gpu_device is not None:
        gpu_device = args.gpu_device.split(",")
        try:
            gpu_device = [int(x) for x in gpu_device]
        except TypeError:
            print("Please provide a list of GPU devices in Integers")
            sys.exit(1)
    else:
        gpu_device = None
    train_prj = Train_Project(config)
    train_prj.load_train_objs()

    # prepare the dataloader
    train_prj.logger.info("generate training, validation datasets, and the model")
    train_prj.prepare_dataloader()
    print(vars(train_prj))
    trainer = Trainer_ln(train_prj, gpu_device)
    trainer.train()
    print("Finished Training")
    # cleanup()


def main():
    parser = argparse.ArgumentParser(description="")
    reqdarg = parser.add_argument_group("required arguments")
    reqdarg.add_argument(
        "-c", dest="config_yaml", type=str, required=True, help="config yaml diretory"
    )
    optarg = parser.add_argument_group("optional arguments")
    optarg.add_argument(
        "-g", dest="gpu_device", type=str, default="0", help="avaiable gpu device(s)"
    )
    args = parser.parse_args()
    dltrain(args)
    return


if __name__ == "__main__":
    main()
