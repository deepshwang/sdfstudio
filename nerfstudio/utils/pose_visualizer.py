import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Referenced from https://github.com/demul/extrinsic2pyramid
class CameraPoseVisualizer:
    def __init__(self, xlim=None, ylim=None, zlim=None, elev_azim=None, colormap="rainbow"):
        self.fig = plt.figure(figsize=(18, 7))
        self.ax = self.fig.gca(projection="3d")
        self.ax.set_aspect("auto")
        if xlim is not None:
            self.ax.set_xlim([xlim[0], xlim[1]])
        if ylim is not None:
            self.ax.set_ylim([ylim[0], ylim[1]])
        if zlim is not None:
            self.ax.set_zlim([zlim[0], zlim[1]])

        if elev_azim is not None:
            assert len(elev_azim) == 2, "Please give 2 values only: elevation / azimuth for view angle!"
            self.ax.view_init(elev_azim[0], elev_azim[1])
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        self.colormap = colormap

    def extrinsic2pyramid(self, extrinsic, color="r", focal_len_scaled=5, aspect_ratio=0.3):
        vertex_std = np.array(
            [
                [0, 0, 0, 1],
                [focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [-focal_len_scaled * aspect_ratio, focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
                [-focal_len_scaled * aspect_ratio, -focal_len_scaled * aspect_ratio, focal_len_scaled, 1],
            ]
        )
        vertex_transformed = vertex_std @ extrinsic.T
        meshes = [
            [vertex_transformed[0, :-1], vertex_transformed[1][:-1], vertex_transformed[2, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[2, :-1], vertex_transformed[3, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[3, :-1], vertex_transformed[4, :-1]],
            [vertex_transformed[0, :-1], vertex_transformed[4, :-1], vertex_transformed[1, :-1]],
            [
                vertex_transformed[1, :-1],
                vertex_transformed[2, :-1],
                vertex_transformed[3, :-1],
                vertex_transformed[4, :-1],
            ],
        ]
        if isinstance(color, str):
            self.ax.add_collection3d(
                Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
            )
        else:
            cmap = mpl.colormaps[self.colormap]
            color = cmap(color)
            self.ax.add_collection3d(
                Poly3DCollection(meshes, facecolors=color, linewidths=0.3, edgecolors=color, alpha=0.35)
            )

    def customize_legend(self, list_label):
        list_handle = []
        for idx, label in enumerate(list_label):
            color = plt.cm.rainbow(idx / len(list_label))
            patch = Patch(color=color, label=label)
            list_handle.append(patch)
        plt.legend(loc="right", bbox_to_anchor=(1.8, 0.5), handles=list_handle)

    def colorbar(self, max_frame_length):
        # cmap = mpl.cm.rainbow
        cmap = mpl.colormaps[self.colormap]
        norm = mpl.colors.Normalize(vmin=0, vmax=max_frame_length)
        self.fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), orientation="vertical", label="Frame Number")

    def add_points(self, rgb3d, xyz3d):
        for c, p in zip(rgb3d, xyz3d):
            self.ax.scatter(p[0], p[1], p[2], s=0.2, color=c / 255)

    def save(self, fname):
        plt.savefig(fname)
