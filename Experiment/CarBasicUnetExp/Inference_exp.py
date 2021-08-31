import json
import os
from SemanticSegmentation.engine.core import Core


def main():
    data = open(os.path.join("config_seg_inf.json"))
    data = json.load(data)
    initialize = Core(data)
    initialize.inference()


if __name__ == "__main__":
    main()
