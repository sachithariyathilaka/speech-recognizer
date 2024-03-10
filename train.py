import sys

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from module.model import SpeechRecognition
from module.speech_module import SpeechModule


class Train:
    def __init__(self, log_path, train_json_path, valid_json_path):
        h_params = SpeechRecognition.hyper_parameters
        h_params.update({})
        model = SpeechRecognition(**h_params)
        speech_module = SpeechModule(model, train_json_path, valid_json_path)
        logger = TensorBoardLogger(log_path, name='speech_recognition')

        trainer = Trainer(
            max_epochs=10,
            num_nodes=1,
            logger=logger,
            gradient_clip_val=1.0,
            val_check_interval=44
        )
        trainer.fit(speech_module)


if __name__ == "__main__":
    log_path = "D:/dataset/"
    train_json_path = "D:/dataset/train.json"
    valid_json_path = "D:/dataset/test.json"

    sys.setrecursionlimit(1000000000)
    Train(log_path, train_json_path, valid_json_path)
