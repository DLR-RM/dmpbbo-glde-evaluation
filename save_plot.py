from pathlib import Path

from matplotlib import pyplot as plt


def save_plot(filename, image_format=".svg", **kwargs):
    directory = Path(kwargs.get("directory", "plots"))

    if ".py" in filename:
        filename = Path(filename).name.replace(".py", image_format)

    directory.mkdir(parents=True, exist_ok=True)
    abs_filename = Path(directory, filename)

    print(f"Saving plot to {abs_filename}")
    # plt.gcf().tight_layout()
    plt.savefig(abs_filename)
