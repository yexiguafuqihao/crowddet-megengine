import numpy as np
from megengine import Tensor
import megengine.functional as F
import pdb
class AnchorGenerator():
    """default anchor generator for fpn.
    This class generate anchors by feature map in level.
    """
    def __init__(self, base_size=16, ratios=[0.5, 1, 2],
        base_scale=2):
        self.base_size = base_size
        self.base_scale = np.array(base_scale)
        self.anchor_ratios = ratios

    def _whctrs(self, anchor):
        """convert anchor box into (w, h, ctr_x, ctr_y)
        """
        w = anchor[:, 2] - anchor[:, 0] + 1
        h = anchor[:, 3] - anchor[:, 1] + 1
        x_ctr = anchor[:, 0] + 0.5 * (w - 1)
        y_ctr = anchor[:, 1] + 0.5 * (h - 1)
        return w, h, x_ctr, y_ctr

    def get_plane_anchors(self, anchor_scales: np.ndarray):
        """get anchors per location on feature map.
        The anchor number is anchor_scales x anchor_ratios
        """
        base_anchor = Tensor([0, 0, self.base_size - 1, self.base_size - 1])
        base_anchor = base_anchor.reshape(1, -1)
        w, h, x_ctr, y_ctr = self._whctrs(base_anchor)
        # ratio enumerate
        size = w * h
        size_ratios = size / self.anchor_ratios
        #pdb.set_trace()
        ws = F.sqrt(size_ratios)
        hs = ws * self.anchor_ratios
        # ws = size_ratios.sqrt().round()
        # hs = (ws * self.anchor_ratios).round()
        # scale enumerate
        anchor_scales = anchor_scales.reshape(1, -1).astype(np.float32)
        ws = F.expand_dims(ws, 1)
        hs = F.expand_dims(hs, 1)
        ws = (ws * anchor_scales).reshape(-1, 1)
        hs = (hs * anchor_scales).reshape(-1, 1)
        # make anchors
        anchors = F.concat(
            [
                x_ctr - 0.5 * (ws - 1),
                y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1),
                y_ctr + 0.5 * (hs - 1),
            ],
            axis=1,
        )
        return anchors.astype(np.float32)

    def get_center_offsets(self, featmap, stride):

        # f_shp = featmap.shape
        # fm_height, fm_width = f_shp[-2], f_shp[-1]
        fm_height, fm_width = featmap.shape[2:]
        shift_x = F.linspace(0, fm_width - 1, fm_width) * stride
        shift_y = F.linspace(0, fm_height - 1, fm_height) * stride

        # make the mesh grid of shift_x and shift_y
        mesh_shape = (fm_height, fm_width)
        
        broad_shift_x = F.broadcast_to(shift_x.reshape(1, -1), mesh_shape)
        broad_shift_y = F.broadcast_to(shift_y.reshape(-1, 1), mesh_shape)
        # broad_shift_x = shift_x.reshape(-1, shift_x.shape[0]).broadcast_to(*mesh_shape)
        # broad_shift_y = shift_y.reshape(shift_y.shape[0], -1).broadcast_to(*mesh_shape)

        flatten_shift_x = broad_shift_x.flatten()
        flatten_shift_y = broad_shift_y.flatten()
        shifts = F.stack([flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y], axis=1)
        # flatten_shift_x = F.add_axis(broad_shift_x.reshape(-1), 1)
        # flatten_shift_y = F.add_axis(broad_shift_y.reshape(-1), 1)

        # shifts = F.concat(
        #     [flatten_shift_x, flatten_shift_y, flatten_shift_x, flatten_shift_y,],
        #     axis=1)
        return shifts 

    def get_anchors_by_feature(self, featmap, stride):
        # shifts shape: [A, 4]
        shifts = self.get_center_offsets(featmap, stride)
        # plane_anchors shape: [B, 4], e.g. B=3
        plane_anchors = self.get_plane_anchors(self.base_scale * stride)
        # all_anchors = shifts.repeat(1,3) + cell_anchors.flatten()
        all_anchors = F.expand_dims(plane_anchors, 0) + F.expand_dims(shifts, 1)
        all_anchors = all_anchors.reshape(-1, 4)
        return all_anchors

    def __call__(self, featmap, stride):
        return self.get_anchors_by_feature(featmap, stride)

