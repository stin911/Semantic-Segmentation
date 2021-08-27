import json
from SemanticSegmentation.engine.core import Core


def main():
    data = open("/SemanticSegmentation/Experiment/config_seg_.json")
    data = json.load(data)
    initialize = Core(data)
    initialize.start_train()


if __name__ == "__main__":
    main()
