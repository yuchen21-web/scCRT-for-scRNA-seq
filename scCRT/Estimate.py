import numpy as np
import torch

from pcurve import PrincipalCurve
# from pcurvepy2 import PrincipalCurve
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.interpolate import interp1d, UnivariateSpline
from sklearn.neighbors import KernelDensity
from collections import deque
from tqdm import tqdm
from scCRTUtils import *



# The main code comes from Slingshot
class Estimate_time():
    def __init__(self, data, cell_labels, start_node=0, k=5, k_p=4, norm=0):

        self.data = data
        self.cell_labels = cell_labels.astype(int)

        self.cluster_labels_onehot = np.zeros((self.cell_labels.shape[0], self.cell_labels.max() + 1))
        self.cluster_labels_onehot[np.arange(self.cell_labels.shape[0]), self.cell_labels] = 1

        self.k = k
        self.k_p = k_p
        self.num_clusters = self.cell_labels.max() + 1
        self.start_node = start_node
        cluster_centres = [data[self.cell_labels == k].mean(axis=0) for k in range(self.num_clusters)]
        self.cluster_centres = np.stack(cluster_centres)
        self.Lineage_class = None      # list of Lineages
        self.cluster_lineages = None # lineages belonging to each cluster
        self.curves = None   # list of principle curves len = #lineages
        self.cell_weights = None  # weights indicating cluster assignments
        self.distances = None
        self.branch_clusters = None
        self.norm = norm
        self.kernel_x = np.linspace(-3, 3, 512)
        kde = KernelDensity(bandwidth=1., kernel='gaussian')
        kde.fit(np.zeros((self.kernel_x.shape[0], 1)))
        self.kernel_y = np.exp(kde.score_samples(self.kernel_x.reshape(-1, 1)))

    def load_params(self, filepath):
        if self.curves is None:
            self.get_lineages()
        params = np.load(filepath, allow_pickle=True).item()
        self.curves = params['curves']   # list of principle curves len = #lineages
        self.cell_weights = params['cell_weights']  # weights indicating cluster assignments
        self.distances = params['distances']

    def save_params(self, filepath):
        params = dict(
            curves=self.curves,
            cell_weights=self.cell_weights,
            distances=self.distances
        )
        np.save(filepath, params)

    def scale_to_range(self, x, a=0, b=1):
        return ((x - x.min()) / (x.max() - x.min())) * (b - a) + a

    def get_lineages(self):
        self.clusters_links = getInnerClusterKNN(self.cell_labels, self.data, k=self.k, norm=self.norm)

        self.Lineage_class, self.branch_clusters = getLineage(self.clusters_links, self.cell_labels, start_node=self.start_node)
        self.cluster_lineages = self.Lineage_class.cluster_lineages

        points_percent, points_Type, cell_belong_point = get_cell_lineages_info(self.Lineage_class, self.data,
                                                                                self.cell_labels, sep_points=99)
        self.cell_belong_type = points_Type[cell_belong_point]

    def fit(self, num_epochs=10):
        if self.curves is None:
            self.get_lineages()
            print('Lineages:', self.Lineage_class.lineages)
            self.construct_initial_curves()
            self.cell_weights = [self.cluster_labels_onehot[:, self.Lineage_class.lineages[l].lineage].sum(axis=1)
                                 for l in range(len(self.Lineage_class.lineages))]
            self.cell_weights = np.stack(self.cell_weights, axis=1)

        for epoch in range(num_epochs):
            # Calculate cell weights
            # cell weight is a matrix #cells x #lineages indicating cell-lineage assignment
            self.calculate_cell_weights()

            # Fit principal curve for all lineages using existing curves
            self.fit_lineage_curves()

            # Ensure starts at 0
            for l_idx, lineage in enumerate(self.Lineage_class.lineages):
                curve = self.curves[l_idx]
                min_time = np.min(curve.pseudotimes_interp[self.cell_weights[:, l_idx] > 0])
                curve.pseudotimes_interp -= min_time

            # Determine average curves
            shrinkage_percentages, cluster_children, cluster_avg_curves = \
                self.avg_curves()

            # Shrink towards average curves in areas of cells common to all branch lineages
            self.shrink_curves(cluster_children, shrinkage_percentages, cluster_avg_curves)


    def construct_initial_curves(self):
        """Constructs lineage principal curves using piecewise linear initialisation"""
        piecewise_linear = list()
        distances = list()
        for l_idx, lineage in enumerate(self.Lineage_class.lineages):
            # Calculate piecewise linear path
            p = np.stack(self.cluster_centres[lineage.lineage])
            s = np.zeros(p.shape[0])  # TODO

            cell_mask = np.logical_or.reduce(
                np.array([self.cell_labels == k for k in lineage.lineage]))

            cells_involved = self.data[cell_mask]

            curve = PrincipalCurve(k=self.k_p)
            curve.project_to_curve(cells_involved, points=p)
            d_sq, dist = curve.project_to_curve(self.data, points=curve.points_interp[curve.order])
            distances.append(d_sq)

            piecewise_linear.append(curve)

        self.curves = piecewise_linear
        self.distances = distances

    def fit_lineage_curves(self):
        assert self.Lineage_class is not None
        assert self.curves is not None
        distances = list()

        # Calculate principal curves
        for l_idx, lineage in enumerate(self.Lineage_class.lineages):
            curve = self.curves[l_idx]

            curve.fit(
                self.data,
                max_iter=1,
                w=self.cell_weights[:, l_idx]
            )

            d_sq, dist = curve.project_to_curve(self.data, curve.points_interp[curve.order])
            distances.append(d_sq)
        self.distances = distances
        # if self.debug_plot_lineages:
        #     self.debug_axes[0, 1].legend()

    def calculate_cell_weights(self):
        """TODO: annotate, this is a translation from R"""
        cell_weights = [self.cluster_labels_onehot[:, self.Lineage_class.lineages[l].lineage].sum(axis=1)
                        for l in range(len(self.Lineage_class.lineages))]
        cell_weights = np.stack(cell_weights, axis=1)

        d_sq = np.stack(self.distances, axis=1)
        d_ord = np.argsort(d_sq, axis=None)
        w_prob = cell_weights/cell_weights.sum(axis=1, keepdims=True)  # shape (cells, lineages)
        w_rnk_d = np.cumsum(w_prob.reshape(-1)[d_ord]) / w_prob.sum()

        z = d_sq
        z_shape = z.shape
        z = z.reshape(-1)
        z[d_ord] = w_rnk_d
        z = z.reshape(z_shape)
        z_prime = 1 - z ** 2
        z_prime[cell_weights == 0] = np.nan
        w0 = cell_weights.copy()
        cell_weights = z_prime / np.nanmax(z_prime, axis=1, keepdims=True) #rowMins(D) / D
        np.nan_to_num(cell_weights, nan=1, copy=False) # handle 0/0
        # cell_weights[is.na(cell_weights)] <- 0
        cell_weights[cell_weights > 1] = 1
        cell_weights[cell_weights < 0] = 0
        cell_weights[w0 == 0] = 0

        reassign = True
        if reassign:
            # add if z < .5
            cell_weights[z < .5] = 1 #(rowMins(D) / D)[idx]

            # drop if z > .9 and cell_weights < .1
            ridx = (z.max(axis=1) > .9) & (cell_weights.min(axis=1) < .1)
            w0 = cell_weights[ridx]
            z0 = z[ridx]
            w0[(z0 > .9) & (w0 < .1)] = 0 # !is.na(Z0) & Z0 > .9 & W0 < .1
            cell_weights[ridx] = w0

        self.cell_weights = cell_weights

    def avg_curves(self):
        """
        Starting at leaves, calculate average curves for each branch

        :return: shrinkage_percentages, cluster_children, cluster_avg_curves
        """
        cell_weights = self.cell_weights
        shrinkage_percentages = list()
        cluster_children = dict()  # maps cluster to children
        lineage_avg_curves = dict()
        cluster_avg_curves = dict()
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            # Starting at leaves, find lineages involved in branch
            k = branch_clusters.pop()
            branch_lineages = self.cluster_lineages[k]
            cluster_children[k] = set()
            for l_idx in branch_lineages:  # loop all lineages through branch
                if l_idx in lineage_avg_curves:  # add avg curve
                    curve = lineage_avg_curves[l_idx]
                else:  # or add leaf curve
                    curve = self.curves[l_idx]
                cluster_children[k].add(curve)

            # Calculate the average curve for this branch
            branch_curves = list(cluster_children[k])
            if self.debug_level > 0:
                print(f'Averaging branch @{k} with lineages:', branch_lineages, branch_curves)

            avg_curve = self.avg_branch_curves(branch_curves)
            cluster_avg_curves[k] = avg_curve

            # Calculate shrinkage weights using areas where cells share lineages
            # note that this also captures cells in average curves, since the
            # lineages which are averaged are present in branch_lineages
            common = cell_weights[:, branch_lineages] > 0
            common_mask = common.mean(axis=1) == 1.
            shrinkage_percent = dict()
            for curve in branch_curves:
                shrinkage_percent[curve] = self.shrinkage_percent(curve, common_mask)
            shrinkage_percentages.append(shrinkage_percent)

            # Add avg_curve to lineage_avg_curve for cluster_children
            for l in branch_lineages:
                lineage_avg_curves[l] = avg_curve

        return shrinkage_percentages, cluster_children, cluster_avg_curves

    def shrink_curves(self, cluster_children, shrinkage_percentages, cluster_avg_curves):
        """
        Starting at root, shrink curves for each branch

        Parameters:
            cluster_children:
            shrinkage_percentages:
            cluster_avg_curves:
        :return:
        """
        branch_clusters = self.branch_clusters.copy()
        while len(branch_clusters) > 0:
            # Starting at root, find lineages involves in branch
            k = branch_clusters.popleft()
            shrinkage_percent = shrinkage_percentages.pop()
            # branch_curves = list(cluster_children[k])
            cluster_avg_curve = cluster_avg_curves[k]
            if self.debug_level > 0:
                print(f'Shrinking branch @{k} with curves')

            # Specify the avg curve for this branch
            self.shrink_branch_curves(list(cluster_children[k]), cluster_avg_curve, shrinkage_percent, k)

    def shrink_branch_curves(self, branch_curves, avg_curve, shrinkage_percent, k):
        """
        Shrinks curves through a branch to the average curve.

        :param branch_curves: list of `PrincipalCurve`s associated with the branch.
        :param avg_curve: `PrincipalCurve` for average curve.
        :param shrinkage_percent: percentage shrinkage, in same order as curve.pseudotimes
        """
        num_dims_reduced = branch_curves[0].points_interp.shape[1]
        shrunk_curves = []
        # Go through "child" lineages, shrinking the curves toward the above average
        for curve in branch_curves:  # curve might be an average curve or a leaf curve
            pct = shrinkage_percent[curve]

            s_interp, p_interp, order = curve.unpack_params()
            avg_s_interp, avg_p_interp, avg_order = avg_curve.unpack_params()
            shrunk_curve = np.zeros_like(p_interp)
            for j in range(num_dims_reduced):
                orig = p_interp[order, j]
                avg = np.interp(#interp1d(
                    s_interp[order],
                    avg_s_interp[avg_order],     # x
                    avg_p_interp[avg_order, j])#,  # y
                shrunk_curve[:, j] = (avg * pct + orig * (1 - pct))

            shrunk_curves.append(shrunk_curve)

        for curve, shrunk_curve in zip(branch_curves, shrunk_curves):
            curve.project_to_curve(self.data, points=shrunk_curve)


    def shrinkage_percent(self, curve, common_ind):
        """Determines how much to shrink a curve"""
        # pst <- crv$lambda
        # pts2wt <- pst
        s_interp, order = curve.pseudotimes_interp, curve.order
        # Cosine kernel quartiles:
        x = self.kernel_x
        y = self.kernel_y
        y = (y.sum() - np.cumsum(y)) / sum(y)
        q1 = np.percentile(s_interp[common_ind], 25)
        q3 = np.percentile(s_interp[common_ind], 75)
        a = q1 - 1.5 * (q3 - q1)
        b = q3 + 1.5 * (q3 - q1)
        x = self.scale_to_range(x, a=a, b=b)
        if q1 == q3:
            pct_l = np.zeros(s_interp.shape[0])
        else:
            pct_l = np.interp(
                s_interp[order],
                x, y
            )

        return pct_l

    def avg_branch_curves(self, branch_curves):
        """branch_lineages is a list of lineages passing through branch"""
        # s_interps, p_interps, orders
        num_cells = branch_curves[0].points_interp.shape[0]
        num_dims_reduced = branch_curves[0].points_interp.shape[1]

        # 1. Interpolate all the lineages over the shared time domain
        branch_s_interps = np.stack([c.pseudotimes_interp for c in branch_curves], axis=1)
        max_shared_pseudotime = branch_s_interps.max(axis=0).min()  # take minimum of maximum pseudotimes for each lineage
        combined_pseudotime = np.linspace(0, max_shared_pseudotime, num_cells)
        curves_dense = list()
        for curve in branch_curves:
            lineage_curve = np.zeros((combined_pseudotime.shape[0], num_dims_reduced))
            order = curve.order
            # Linearly interpolate each dimension as a function of pseudotime
            for j in range(num_dims_reduced):
                lin_interpolator = interp1d(
                    curve.pseudotimes_interp[order], # x
                    curve.points_interp[order, j],   # y
                    assume_sorted=True
                )
                lineage_curve[:, j] = lin_interpolator(combined_pseudotime)
            curves_dense.append(lineage_curve)

        curves_dense = np.stack(curves_dense, axis=1)  # (n, L_b, J)

        # 2. Average over these curves and project the data onto the result
        avg = curves_dense.mean(axis=1)  # avg is already "sorted"
        avg_curve = PrincipalCurve(k=self.k_p)
        avg_curve.project_to_curve(self.data, points=avg)
        return avg_curve

    @property
    def unified_pseudotime(self):
        pseudotime = np.zeros_like(self.curves[0].pseudotimes_interp)
        for l_idx, lineage in enumerate(self.Lineage_class.lineages):
            curve = self.curves[l_idx]
            cell_mask = np.logical_or.reduce(
                np.array([self.cell_labels == k for k in lineage.lineage]))
            pseudotime[cell_mask] = curve.pseudotimes_interp[cell_mask]
        return pseudotime

    def list_lineages(self, cluster_to_label):
        for lineage in self.Lineage_class.lineages:
            print(', '.join([
                cluster_to_label[l] for l in lineage.lineage
            ]))

