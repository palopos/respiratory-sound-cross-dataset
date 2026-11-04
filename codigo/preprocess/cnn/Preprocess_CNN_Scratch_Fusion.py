import os
import sys

_SCRIPT_DIR = os.path.normpath(os.path.dirname(__file__) or ".")
sys.path.insert(0, os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "train", "cnn")))

import Train_CNN_Scratch_Fusion as pipeline


def main():
    target_width = pipeline.run_preprocessing()
    print(f">>> Preprocesado fusionado completado. target_width={target_width}")


if __name__ == "__main__":
    main()
