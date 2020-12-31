import matplotlib.pyplot as plt


def show_images(**images):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(20, 12))

    for i in range(len(images)):
        axes[0].imshow(images)

    fig.tight_layout()
