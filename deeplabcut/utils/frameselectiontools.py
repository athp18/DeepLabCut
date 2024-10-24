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
"""
DeepLabCut2.0 Toolbox (deeplabcut.org)
© A. & M. Mathis Labs
https://github.com/DeepLabCut/DeepLabCut
Please see AUTHORS for contributors.

https://github.com/DeepLabCut/DeepLabCut/blob/master/AUTHORS
Licensed under GNU Lesser General Public License v3.0
"""


import math

import cv2
import numpy as np
from skimage.util import img_as_ubyte
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture
from tqdm import tqdm


def GMMbasedFrameselection(
    clip,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    max_iter=100,
    color=False,
):
    """
    This function selects frames from a video clip using Gaussian Mixture Model clustering.
    Each frame is treated as a data point, and frames from different clusters are selected.
    This ensures that the selected frames represent diverse visual content.

    Parameters
    ----------
    clip : VideoFileClip object
        The video clip from which to extract frames.

    numframes2pick : int
        The number of frames to select.

    start : float
        The starting fraction of the video duration (between 0 and 1).

    stop : float
        The ending fraction of the video duration (between 0 and 1).

    Index : list or ndarray, optional
        A list of frame indices to consider. If None, frames are sampled based on start and stop.

    step : int, default=1
        Step size for sampling frames.

    resizewidth : int, default=30
        The width to which frames are resized (aspect ratio is maintained).

    max_iter : int, default=100
        Maximum number of iterations for the Gaussian Mixture Model.

    color : bool, default=False
        Whether to include color information in clustering.

    Returns
    -------
    frames2pick : list
        List of selected frame indices.
    """
    print(
        "GMM-based extracting of frames from",
        round(start * clip.duration, 2),
        "seconds to",
        round(stop * clip.duration, 2),
        "seconds.",
    )
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index >= startindex) & (Index <= stopindex)]  # Crop to range

    nframes = len(Index)
    if nframes < numframes2pick:
        print("Not enough frames to pick from. Returning all available frames.")
        return list(Index)

    clipresized = clip.resize(width=resizewidth)
    ny, nx = clipresized.size
    frame0 = img_as_ubyte(clip.get_frame(0))
    ncolors = frame0.shape[2] if frame0.ndim == 3 else 1

    print("Extracting and downsampling...", nframes, "frames from the video.")

    # Initialize data array
    if color and ncolors > 1:
        DATA = np.zeros((nframes, nx * ny * 3), dtype=np.uint8)
    else:
        DATA = np.zeros((nframes, nx * ny), dtype=np.uint8)

    for counter, index in tqdm(enumerate(Index), total=nframes):
        frame_time = index / clip.fps
        frame = img_as_ubyte(clipresized.get_frame(frame_time))
        if color and ncolors > 1:
            DATA[counter, :] = frame.reshape(-1)
        else:
            if ncolors == 1:
                gray_frame = frame
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            DATA[counter, :] = gray_frame.flatten()

    print("Fitting Gaussian Mixture Model... (this might take a while)")
    gmm = GaussianMixture(
        n_components=numframes2pick, max_iter=max_iter, random_state=42
    )
    gmm.fit(DATA)

    labels = gmm.predict(DATA)
    frames2pick = []
    for clusterid in range(numframes2pick):
        cluster_indices = np.where(labels == clusterid)[0]
        if len(cluster_indices) > 0:
            selected_index = np.random.choice(cluster_indices)
            frames2pick.append(Index[selected_index])

    clipresized.close()
    del clipresized
    return frames2pick


def GMMbasedFrameselectioncv2(
    cap,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    max_iter=100,
    color=False,
):
    """
    This function selects frames from a video capture object using Gaussian Mixture Model clustering.
    Each frame is treated as a data point, and frames from different clusters are selected.

    Parameters
    ----------
    cap : VideoCapture object
        The OpenCV video capture object.

    numframes2pick : int
        The number of frames to select.

    start : float
        The starting fraction of the video duration (between 0 and 1).

    stop : float
        The ending fraction of the video duration (between 0 and 1).

    Index : list or ndarray, optional
        A list of frame indices to consider. If None, frames are sampled based on start and stop.

    step : int, default=1
        Step size for sampling frames.

    resizewidth : int, default=30
        The width to which frames are resized (aspect ratio is maintained).

    max_iter : int, default=100
        Maximum number of iterations for the Gaussian Mixture Model.

    color : bool, default=False
        Whether to include color information in clustering.

    Returns
    -------
    frames2pick : list
        List of selected frame indices.
    """
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = resizewidth / width
    if ratio > 1:
        raise ValueError("resizewidth is larger than original frame width.")

    print(
        "GMM-based extracting of frames from",
        round(start * nframes / fps, 2),
        "seconds to",
        round(stop * nframes / fps, 2),
        "seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index >= startindex) & (Index <= stopindex)]  # Crop to range

    nframes = len(Index)
    if nframes < numframes2pick:
        print("Not enough frames to pick from. Returning all available frames.")
        return list(Index)

    ny = int(height * ratio)
    nx = int(width * ratio)

    # Initialize data array
    if color:
        DATA = np.zeros((nframes, ny * nx * 3), dtype=np.uint8)
    else:
        DATA = np.zeros((nframes, ny * nx), dtype=np.uint8)

    print("Extracting and downsampling...", nframes, "frames from the video.")

    for counter, idx in tqdm(enumerate(Index), total=nframes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, (nx, ny), interpolation=cv2.INTER_AREA)
        if color:
            DATA[counter, :] = frame_resized.reshape(-1)
        else:
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            DATA[counter, :] = gray_frame.flatten()

    print("Fitting Gaussian Mixture Model... (this might take a while)")
    gmm = GaussianMixture(
        n_components=numframes2pick, max_iter=max_iter, random_state=42
    )
    gmm.fit(DATA)

    labels = gmm.predict(DATA)
    frames2pick = []
    for clusterid in range(numframes2pick):
        cluster_indices = np.where(labels == clusterid)[0]
        if len(cluster_indices) > 0:
            selected_index = np.random.choice(cluster_indices)
            frames2pick.append(Index[selected_index])

    return frames2pick


def EMFrameSelection(
    cap,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    max_iter=100,
    cov_mat_type=cv2.ml.EM_COV_MAT_DIAGONAL,
    color=False,
):
    """
    This function selects frames from a video capture object using the EM algorithm for Gaussian Mixture Models.
    Each frame is treated as a data point, and frames from different clusters are selected.

    Parameters
    ----------
    cap : cv2.VideoCapture object
        The OpenCV video capture object.

    numframes2pick : int
        The number of frames to select.

    start : float
        The starting fraction of the video duration (between 0 and 1).

    stop : float
        The ending fraction of the video duration (between 0 and 1).

    Index : list or ndarray, optional
        A list of frame indices to consider. If None, frames are sampled based on start and stop.

    step : int, default=1
        Step size for sampling frames.

    resizewidth : int, default=30
        The width to which frames are resized (aspect ratio is maintained).

    max_iter : int, default=100
        Maximum number of iterations for the EM algorithm.

    cov_mat_type : int, default=cv2.ml.EM_COV_MAT_DIAGONAL
        Type of the covariance matrices for the GMM components.
        Options include cv2.ml.EM_COV_MAT_SPHERICAL, cv2.ml.EM_COV_MAT_DIAGONAL, cv2.ml.EM_COV_MAT_GENERIC.

    color : bool, default=False
        Whether to include color information in clustering.

    Returns
    -------
    frames2pick : list
        List of selected frame indices.
    """
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = resizewidth / width
    if ratio > 1:
        raise ValueError("resizewidth is larger than original frame width.")

    print(
        "EM-based extracting of frames from",
        round(start * nframes / fps, 2),
        "seconds to",
        round(stop * nframes / fps, 2),
        "seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index >= startindex) & (Index <= stopindex)]  # Crop to range

    nframes = len(Index)
    if nframes < numframes2pick:
        print("Not enough frames to pick from. Returning all available frames.")
        return list(Index)

    ny = int(height * ratio)
    nx = int(width * ratio)

    if color:
        DATA = np.zeros((nframes, ny * nx * 3), dtype=np.float32)
    else:
        DATA = np.zeros((nframes, ny * nx), dtype=np.float32)

    print("Extracting and downsampling...", nframes, "frames from the video.")

    for counter, idx in tqdm(enumerate(Index), total=nframes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_resized = cv2.resize(frame, (nx, ny), interpolation=cv2.INTER_AREA)
        if color:
            DATA[counter, :] = frame_resized.flatten()
        else:
            gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            DATA[counter, :] = gray_frame.flatten()

    # Normalize data
    DATA -= np.mean(DATA, axis=0)

    print("Fitting EM Gaussian Mixture Model... (this might take a while)")

    # Initialize EM model
    em = cv2.ml.EM_create()
    em.setClustersNumber(numframes2pick)
    em.setCovarianceMatrixType(cov_mat_type)
    em.setTermCriteria(
        (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, max_iter, 1e-6)
    )

    # Train EM model
    retval, logLikelihoods, labels, probs = em.trainEM(DATA)

    labels = labels.flatten().astype(int)
    frames2pick = []
    for clusterid in range(numframes2pick):
        cluster_indices = np.where(labels == clusterid)[0]
        if len(cluster_indices) > 0:
            selected_index = np.random.choice(cluster_indices)
            frames2pick.append(Index[selected_index])

    return frames2pick


def UniformFrames(clip, numframes2pick, start, stop, Index=None):
    """Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    print(
        "Uniformly extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(clip.duration * clip.fps * stop),
                size=numframes2pick,
                replace=False,
            )
        else:
            frames2pick = np.random.choice(
                range(
                    math.floor(start * clip.duration * clip.fps),
                    math.ceil(clip.duration * clip.fps * stop),
                ),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(clip.fps * clip.duration * start))
        stopindex = int(np.ceil(clip.fps * clip.duration * stop))
        Index = np.array(Index, dtype=int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


# uses openCV
def UniformFramescv2(cap, numframes2pick, start, stop, Index=None):
    """Temporally uniformly sampling frames in interval (start,stop).
    Visual information of video is irrelevant for this method. This code is fast and sufficient (to extract distinct frames),
    when behavioral videos naturally covers many states.

    The variable Index allows to pass on a subindex for the frames.
    """
    nframes = len(cap)
    print(
        "Uniformly extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )

    if Index is None:
        if start == 0:
            frames2pick = np.random.choice(
                math.ceil(nframes * stop), size=numframes2pick, replace=False
            )
        else:
            frames2pick = np.random.choice(
                range(math.floor(nframes * start), math.ceil(nframes * stop)),
                size=numframes2pick,
                replace=False,
            )
        return frames2pick
    else:
        startindex = int(np.floor(nframes * start))
        stopindex = int(np.ceil(nframes * stop))
        Index = np.array(Index, dtype=int)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!
        if len(Index) >= numframes2pick:
            return list(np.random.permutation(Index)[:numframes2pick])
        else:
            return list(Index)


def KmeansbasedFrameselection(
    clip,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.

    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick."""

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * clip.duration, 2),
        " seconds to",
        round(stop * clip.duration, 2),
        " seconds.",
    )
    startindex = int(np.floor(clip.fps * clip.duration * start))
    stopindex = int(np.ceil(clip.fps * clip.duration * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = int(nframes / 2)

    if len(Index) >= numframes2pick:
        clipresized = clip.resize(width=resizewidth)
        ny, nx = clipresized.size
        frame0 = img_as_ubyte(clip.get_frame(0))
        if np.ndim(frame0) == 3:
            ncolors = np.shape(frame0)[2]
        else:
            ncolors = 1
        print("Extracting and downsampling...", nframes, " frames from the video.")

        if color and ncolors > 1:
            DATA = np.zeros((nframes, nx * 3, ny))
            for counter, index in tqdm(enumerate(Index)):
                image = img_as_ubyte(
                    clipresized.get_frame(index * 1.0 / clipresized.fps)
                )
                DATA[counter, :, :] = np.vstack(
                    [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                )
        else:
            DATA = np.zeros((nframes, nx, ny))
            for counter, index in tqdm(enumerate(Index)):
                if ncolors == 1:
                    DATA[counter, :, :] = img_as_ubyte(
                        clipresized.get_frame(index * 1.0 / clipresized.fps)
                    )
                else:  # attention: averages over color channels to keep size small / perhaps you want to use color information?
                    DATA[counter, :, :] = img_as_ubyte(
                        np.array(
                            np.mean(
                                clipresized.get_frame(index * 1.0 / clipresized.fps), 2
                            ),
                            dtype=np.uint8,
                        )
                    )

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )

        clipresized.close()
        del clipresized
        return list(np.array(frames2pick))
    else:
        return list(Index)


def KmeansbasedFrameselectioncv2(
    cap,
    numframes2pick,
    start,
    stop,
    Index=None,
    step=1,
    resizewidth=30,
    batchsize=100,
    max_iter=50,
    color=False,
):
    """This code downsamples the video to a width of resizewidth.
    The video is extracted as a numpy array, which is then clustered with kmeans, whereby each frames is treated as a vector.
    Frames from different clusters are then selected for labeling. This procedure makes sure that the frames "look different",
    i.e. different postures etc. On large videos this code is slow.

    Consider not extracting the frames from the whole video but rather set start and stop to a period around interesting behavior.

    Note: this method can return fewer images than numframes2pick.

    Attention: the flow of commands was not optimized for readability, but rather speed. This is why it might appear tedious and repetitive.
    """
    nframes = len(cap)
    nx, ny = cap.dimensions
    ratio = resizewidth * 1.0 / nx
    if ratio > 1:
        raise Exception("Choice of resizewidth actually upsamples!")

    print(
        "Kmeans-quantization based extracting of frames from",
        round(start * nframes * 1.0 / cap.fps, 2),
        " seconds to",
        round(stop * nframes * 1.0 / cap.fps, 2),
        " seconds.",
    )
    startindex = int(np.floor(nframes * start))
    stopindex = int(np.ceil(nframes * stop))

    if Index is None:
        Index = np.arange(startindex, stopindex, step)
    else:
        Index = np.array(Index)
        Index = Index[(Index > startindex) * (Index < stopindex)]  # crop to range!

    nframes = len(Index)
    if batchsize > nframes:
        batchsize = nframes // 2

    ny_ = np.round(ny * ratio).astype(int)
    nx_ = np.round(nx * ratio).astype(int)
    DATA = np.empty((nframes, ny_, nx_ * 3 if color else nx_))
    if len(Index) >= numframes2pick:
        if (
            np.mean(np.diff(Index)) > 1
        ):  # then non-consecutive indices are present, thus cap.set is required (which slows everything down!)
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    cap.set_to_frame(index)  # extract a particular frame
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)
        else:
            print("Extracting and downsampling...", nframes, " frames from the video.")
            if color:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.hstack(
                            [image[:, :, 0], image[:, :, 1], image[:, :, 2]]
                        )
            else:
                for counter, index in tqdm(enumerate(Index)):
                    frame = cap.read_frame(crop=True)
                    if frame is not None:
                        image = img_as_ubyte(
                            cv2.resize(
                                frame,
                                None,
                                fx=ratio,
                                fy=ratio,
                                interpolation=cv2.INTER_NEAREST,
                            )
                        )  # color trafo not necessary; lack thereof improves speed.
                        DATA[counter, :, :] = np.mean(image, 2)

        print("Kmeans clustering ... (this might take a while)")
        data = DATA - DATA.mean(axis=0)
        data = data.reshape(nframes, -1)  # stacking

        kmeans = MiniBatchKMeans(
            n_clusters=numframes2pick, tol=1e-3, batch_size=batchsize, max_iter=max_iter
        )
        kmeans.fit(data)
        frames2pick = []
        for clusterid in range(numframes2pick):  # pick one frame per cluster
            clusterids = np.where(clusterid == kmeans.labels_)[0]

            numimagesofcluster = len(clusterids)
            if numimagesofcluster > 0:
                frames2pick.append(
                    Index[clusterids[np.random.randint(numimagesofcluster)]]
                )
        # cap.release() >> still used in frame_extraction!
        return list(np.array(frames2pick))
    else:
        return list(Index)
