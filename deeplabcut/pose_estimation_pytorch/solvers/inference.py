#
# DeepLabCut Toolbox (deeplabcut.org)
# © A. & M.W. Mathis Labs
# https://github.com/DeepLabCut/DeepLabCut
#
# Please see AUTHORS for contributors.
# https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
#
# Licensed under GNU Lesser General Public License v3.0
#

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deeplabcut.pose_estimation_tensorflow.lib.inferenceutils import (
    Assembly,
    evaluate_assembly,
)


# TODO: DEPRECATED
def get_prediction(
    cfg: dict, output: Tuple[np.ndarray, np.ndarray], stride: int = 8
) -> np.ndarray:
    """Generates pose predictions from the model outputwhich is a tuple given by (heatmaps,location refinement fields)).

    It uses the predicted heatmaps to estimate the keypoints' locations and applies location refinement if enabled.
    Refer to: https://www.nature.com/articles/s41592-022-01443-0 for more information about the overall process.

    Args:
        cfg: config file in dict
        output: heatmaps, locref
            heatmaps: the probability that a
                keypoint occurs at a particular location
            locref: location refinement fields
                that predict offsets to mitigate quantization errors due to downsampled score maps
        stride: window stride; defaults to 8, Optional

    Returns:
        Array of poses

    Examples:
        >>> # Define the cfg dictionary and model output
        >>> cfg = {'location_refinement': True, 'locref_stdev': 0.1}
        >>> heatmaps = np.random.rand(1, 17, 128, 128)
        >>> locref = np.random.rand(1, 17, 128, 128)
        >>> output = (heatmaps, locref)
        >>> # Get the predicted poses
        >>> poses = get_prediction(cfg, output)
    """

    poses = []
    heatmaps, locref = output
    heatmaps = nn.Sigmoid()(heatmaps)
    heatmaps = heatmaps.permute(0, 2, 3, 1).detach().cpu().numpy()
    locref = locref.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(heatmaps.shape[0]):
        shape = locref[i].shape
        locref_i = np.reshape(locref, (shape[0], shape[1], -1, 2))
        if cfg["location_refinement"]:
            locref_i = locref_i * cfg["locref_stdev"]
        pose = multi_pose_predict(heatmaps[i], locref_i, stride, 1)
        poses.append(pose)

    return np.stack(poses, axis=0)


def get_scores(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    pcutoff: Optional[float] = None,
    bodyparts: List[str] = None,
) -> Dict:
    """Computes for the different scores given the grount truth and the predictions.

    The different scores computed are based on the COCO metrics: https://cocodataset.org/#keypoints-eval
    RMSE (Root Mean Square Error)
    OKS mAP (Mean Average Precision)
    OKS mAR (Mean Average Recall)

    Args:
        prediction: prediction df, should already be matched to ground truth using
            Hungarian Algorithm (Ref: https://brilliant.org/wiki/hungarian-matching/)
        target: ground truth dataframe
        pcutoff: the value used to compute the pcutoff scores
        bodyparts: names of the bodyparts. Defaults to None.

    Returns:
        scores: A dictionary of scores containign the following keys:
                      ['rmse', 'rmse_pcutoff', 'mAP', 'mAR', 'mAP_pcutoff', 'mAR_pcutoff']

    Examples:
        >>> # Define the p-cutoff, prediction, and target DataFrames
        >>> pcutoff = 0.5
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the scores
        >>> scores = get_scores(prediction, target, pcutoff)
        >>> print(scores)
        {
            'rmse': 0.156,
            'rmse_pcutoff': 0.115,
            'mAP': 84.2,
            'mAR': 74.5,
            'mAP_pcutoff': 91.3,
            'mAR_pcutoff': 82.5
        }  # Sample output scores
    """
    if pcutoff is None:
        pcutoff = -1

    rmse, rmse_p = get_rmse(prediction, target, pcutoff=pcutoff, bodyparts=bodyparts)
    oks, oks_p = get_oks(prediction, target, pcutoff=pcutoff, bodyparts=bodyparts)
    return {
        "rmse": np.nanmean(rmse),
        "rmse_pcutoff": np.nanmean(rmse_p),
        "mAP": 100 * oks["mAP"],
        "mAR": 100 * oks["mAR"],
        "mAP_pcutoff": 100 * oks_p["mAP"],
        "mAR_pcutoff": 100 * oks_p["mAR"],
    }


def get_rmse(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    pcutoff: float = -1,
    bodyparts: List[str] = None,
) -> Tuple[float, float]:
    """Computes the root mean square error (rmse) for predictions vs the ground truth labels

    Assumes hungarian algorithm matching (https://brilliant.org/wiki/hungarian-matching/))
    has already be applied to match predicted animals and ground truth ones.

    Args:
        prediction: prediction dataframe
        target: target dataframe
        pcutoff: Confidence lower bound for a keypoint to be considered as detected.
                                    Defaults to -1.
        bodyparts: list of the bodyparts names. Defaults to None.

    Returns:
        rmse: rmse without cutoff
        rmse_p : rmse with cutoff

    Example:
        >>> # Define the prediction and target DataFrames
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the RMSE values
        >>> rmse, rmse_pcutoff = get_rmse(prediction, target, pcutoff=0.5)
        >>> print(rmse, rmse_pcutoff)
        0.145 0.105  # Sample output RMSE values
    """
    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]
    mask = prediction[scorer_pred].xs("likelihood", level=2, axis=1) >= pcutoff
    if bodyparts:
        diff = (
            target[scorer_target][bodyparts] - prediction[scorer_pred][bodyparts]
        ) ** 2
    else:
        diff = (target[scorer_target] - prediction[scorer_pred]) ** 2
    mse = diff.xs("x", level=2, axis=1) + diff.xs("y", level=2, axis=1)
    rmse = np.sqrt(mse)
    rmse_p = np.sqrt(mse[mask])

    return rmse, rmse_p


def get_oks(
    prediction: pd.DataFrame,
    target: pd.DataFrame,
    oks_sigma=0.1,
    margin=0,
    symmetric_kpts=None,
    pcutoff: float = -1,
    bodyparts: List[str] = None,
) -> Tuple[Dict, Dict]:
    """Computes the object keypoint similarity (OKS) scores for predictions.

    OKS is defined in https://cocodataset.org/#keypoints-eval

    Args:
        prediction: prediction dataframe
        target: target dataframe
        oks_sigma: Sigma for oks conputation. Defaults to 0.1.
        margin: margin used for bbox computation. Defaults to 0.
        symmetric_kpts: Not supported yet. Defaults to None.
        pcutoff: Confidence lower bound for a keypoint to be considered as detected.
                                    Defaults to -1.
        bodyparts: list of the bodyparts names. Defaults to None.

    Returns:
        oks_raw: oks scores without p_cutoff
        oks_pcutoff: oks scores with pcutoff

    Examples:
        >>> # Define the prediction and target DataFrames
        >>> prediction = pd.DataFrame(...)  # Your DataFrame here
        >>> target = pd.DataFrame(...)      # Your DataFrame here
        >>> # Compute the OKS scores
        >>> oks, oks_pcutoff = get_oks(prediction, target, oks_sigma=0.2, pcutoff=0.5)
        >>> print(oks, oks_pcutoff)
        {'mAP': 0.842, 'mAR': 0.745} {'mAP': 0.913, 'mAR': 0.825}  # Sample output OKS scores
    """

    scorer_pred = prediction.columns[0][0]
    scorer_target = target.columns[0][0]

    if bodyparts is not None:
        idx_slice = pd.IndexSlice[:, :, bodyparts, :]
        prediction = prediction.loc[:, idx_slice]
        target = target.loc[:, idx_slice]
    mask = prediction[scorer_pred].xs("likelihood", level=2, axis=1) >= pcutoff

    # Convert predictions to DLC assemblies
    assemblies_pred_raw, unique_pred_raw = conv_df_to_assemblies(
        prediction[scorer_pred]
    )
    assemblies_gt_raw, unique_gt_raw = conv_df_to_assemblies(target[scorer_target])

    assemblies_pred_masked, unique_pred_masked = conv_df_to_assemblies(
        prediction[scorer_pred][mask]
    )
    assemblies_gt_masked, unique_gt_masked = conv_df_to_assemblies(
        target[scorer_target][mask]
    )

    oks_assemblies_raw = evaluate_assembly(
        assemblies_pred_raw,
        assemblies_gt_raw,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
    )
    if unique_pred_raw is not None and unique_gt_raw is not None:
        oks_unique_raw = evaluate_assembly(
            unique_pred_raw,
            unique_gt_raw,
            oks_sigma,
            margin=margin,
            symmetric_kpts=symmetric_kpts,
        )

    oks_pcutoff = evaluate_assembly(
        assemblies_pred_masked,
        assemblies_gt_masked,
        oks_sigma,
        margin=margin,
        symmetric_kpts=symmetric_kpts,
    )
    if unique_pred_masked is not None and unique_gt_masked is not None:
        oks_unique_masked = evaluate_assembly(
            unique_pred_masked,
            unique_gt_masked,
            oks_sigma,
            margin=margin,
            symmetric_kpts=symmetric_kpts,
        )

    return oks_assemblies_raw, oks_pcutoff


def conv_df_to_assemblies(df: pd.DataFrame) -> Tuple[dict, Optional[dict]]:
    """
    Convert a dataframe to an assemblies dictionary

    Args:
        df : dataframe of coordinates/predictions,
        df is expected to have a multi_index of shape (num_animals, num_keypoints, 2 or 3)
    Returns:
        assemblies: dictionary of the assemblies of keypoints
        if there are unique bodyparts, a dictionary containing unique bodyparts
    """
    individuals = df.columns.get_level_values(0)
    df_bodyparts = df.loc[:, individuals != "single"]
    assemblies = _df_to_dict(df_bodyparts)

    unique_keypoints = None
    if "single" in individuals:
        df_unique = df.loc[:, individuals == "single"]
        unique_keypoints = _df_to_dict(df_unique)

    return assemblies, unique_keypoints


def _df_to_dict(df: pd.DataFrame) -> dict:
    data = {}
    num_animals = len(df.columns.get_level_values(0).unique())
    num_kpts = len(df.columns.get_level_values(1).unique())
    for image_path in df.index:
        row = df.loc[image_path].to_numpy()
        row = row.reshape(num_animals, num_kpts, -1)

        kpt_lst = []
        for i in range(num_animals):
            ass = Assembly.from_array(row[i])
            if len(ass):
                kpt_lst.append(ass)

        data[image_path] = kpt_lst

    return data


# DEPRECATED
def get_top_values(scmap: np.array, n_top: int = 5) -> Tuple[np.array, np.array]:
    """This function computes for the top n values from a given scoremap.

    Args:
        scmap: score map;
            which encode the probability that a keypoint occurs at a particular location
        n_top: top n elements in the set. Defaults to 5.

    Returns:
        Top n values of in the scoreemap
    """
    batchsize, ny, nx, num_joints = scmap.shape
    scmap_flat = scmap.reshape(batchsize, nx * ny, num_joints)
    if n_top == 1:
        scmap_top = np.argmax(scmap_flat, axis=1)[None]
    else:
        scmap_top = np.argpartition(scmap_flat, -n_top, axis=1)[:, -n_top:]
        for ix in range(batchsize):
            vals = scmap_flat[ix, scmap_top[ix], np.arange(num_joints)]
            arg = np.argsort(-vals, axis=0)
            scmap_top[ix] = scmap_top[ix, arg, np.arange(num_joints)]
        scmap_top = scmap_top.swapaxes(0, 1)

    Y, X = np.unravel_index(scmap_top, (ny, nx))
    return Y, X


# DEPRECATED
def multi_pose_predict(
    scmap: np.array, locref: np.array, stride: int, num_outputs: int
) -> np.array:
    """This function generates the multi pose predictions from the model of the output (heatmaps and loc refinement fields).

    Refer to: https://www.nature.com/articles/s41592-022-01443-0 for more information about the overall process.

    Args:
        scmap: score map; which encode the probability that a keypoint occurs at a particular location
        locref: location refinement fields that predict offsets to mitigate quantization errors due to downsampled score maps
        stride: window stride; defaults to 8
        num_outputs: The expected number of outputs.

    Returns:
        pose: Multi-pose predictions
    """
    Y, X = get_top_values(scmap[None], num_outputs)
    Y, X = Y[:, 0], X[:, 0]
    num_joints = scmap.shape[2]

    DZ = np.zeros((num_outputs, num_joints, 3))
    indices = np.indices((num_outputs, num_joints))
    x = X[indices[0], indices[1]]
    y = Y[indices[0], indices[1]]
    DZ[:, :, :2] = locref[y, x, indices[1], :]
    DZ[:, :, 2] = scmap[y, x, indices[1]]

    X = X.astype("float32") * stride[1] + 0.5 * stride[1] + DZ[:, :, 0]
    Y = Y.astype("float32") * stride[0] + 0.5 * stride[0] + DZ[:, :, 1]
    P = DZ[:, :, 2]

    pose = np.empty((num_joints, num_outputs * 3), dtype="float32")
    pose[:, 0::3] = X.T
    pose[:, 1::3] = Y.T
    pose[:, 2::3] = P.T

    return pose
