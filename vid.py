from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
from tqdm import tqdm


def create_aligned_img(imgs: List[np.ndarray], labels: Optional[List[str]]=None, figsize=None, dpi=100, fontsize=12):
    if figsize is None:
        figsize = (np.array(imgs[0].shape[:2]) / imgs[0].shape[0] * 5).astype(int)

        if labels is not None:
            figsize[1] += 1.5


    fig, axs = plt.subplots(1, len(imgs), figsize=figsize, dpi=dpi)
    for i in range(len(imgs)):
        ax = axs[i]
        im = imgs[i]
        
        ax.imshow(im)
        
        if labels is not None:
            ax.set_xlabel(labels[i], fontsize=fontsize)

        # make spines (the box) invisible
        plt.setp(ax.spines.values(), visible=False)
        # remove ticks and labels for the left axis
        ax.tick_params(left=False, bottom=False, labelbottom=False, labelleft=False)
        
        # ax.set_aspect('auto')

        
    fig.tight_layout(pad=0)
    fig.canvas.draw()
    aligned_im = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    aligned_im = aligned_im.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close()

    return aligned_im


im_dirs = [
    "/rds/project/rds-qxpdOeYWi78/Deformable-3D-Gaussians/output/insta_soubhik/layer/bg_style2_b8/test/ours_40000/gt",
    "/rds/project/rds-qxpdOeYWi78/Deformable-3D-Gaussians/output/insta_soubhik/layer/bg_style2_b8/test/ours_40000/renders",
    ]
N = len(list(Path(im_dirs[0]).glob('*')))
N = 500

img_labels = [
    "GT",
    "RGB",
    # "RGB erode",
]

out_path = './vid.mp4'

writer = imageio.get_writer(str(out_path), quality=8, fps=15)

for i in tqdm(range(N)):
    imgs = []
    for im_dir in im_dirs:
        im_p = sorted(list(Path(im_dir).glob('*.png')), key=lambda x: int(x.stem))[i]
        imgs.append(imageio.imread(str(im_p)))

    im = create_aligned_img(imgs, labels=img_labels, figsize=(10, 4))
    writer.append_data(im)
writer.close()


print(f'Video saved to: {str(out_path)}')
