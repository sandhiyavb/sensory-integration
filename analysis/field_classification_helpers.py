"""
A collection of helper functions.
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as nd
from scipy.spatial.distance import pdist


def classify_modulated_vector(idx_modulated, idx_vector) :
    """ Using classification algorithms for HD modulated fields and ECD fields
    will return redundant results. This function can be used to remove 
    redundant entries. 
    
    Keyword arguments :
    idx_modulated -- list of unit ids that are HD modulated
    idx_vector -- list of unit ids that are ECD like
    """
    mask         = np.isin(idx_modulated, idx_vector)
    mask2        = np.isin(idx_vector, idx_modulated)
    id_modulated = idx_modulated[~mask]         # modulated but not vector
    id_vector    = idx_vector[mask2]            # overlap, in idx_vector's order
    return mask, mask2, id_modulated, id_vector


def reclassify(data_modulated, data_vector) :
    """ Re-classifies data from the classification of HD modulated and ECD fields.
    The reclassification is done in place.
    
    Each entry must be a mutable list (it is modified in place) of the form
    [fields, field_ids, pairwise_distances_centers, pairwise_distances_hd].

    Keyword arguments :
    data_modulated : list of modulated-field entries; on return, units that are
                     also vector-like are removed.
    data_vector    : list of vector-field entries; on return, reduced to the
                     units present in both classifications (the overlap).
    """
    for m,v in zip(data_modulated,data_vector) :
        mask, mask2, ids_mod, ids_vec = classify_modulated_vector(m[1], v[1])
        m[1] = ids_mod
        v[1] = ids_vec
        m[0] = (m[0])[~mask]
        v[0] = (v[0])[mask2]
        
def is_blob(activity_map, threshold=0.15, max_clusters=4,
            max_field_fraction=0.5, verbose=False) :
    """ Decide whether or not a given spatial activity map contains place fields.

    Keyword arguments :
    activity_map       : 2d numpy array of spatial activations
    threshold          : pixels below threshold * peak are zeroed before
                         clustering (fraction of the peak activity)
    max_clusters       : more clusters than this -> not place like
    max_field_fraction : largest cluster bigger than this fraction of the whole
                         map -> too large to be a place field
    verbose            : print diagnostic messages

    Returns (is_field, field), where `field` is the thresholded map containing
    ALL surviving clusters.
    """
    activity_map = nd.gaussian_filter(activity_map, sigma=2)
    activity_map = np.where(activity_map < threshold * activity_map.max(),
                            0, activity_map)
    labels, nb = nd.label(activity_map > 0)

    if nb == 0 :
        if verbose :
            print("no cluster -- not place like")
        return False, labels

    counts = np.bincount(labels.ravel(), minlength=nb + 1)  # counts[0] = background
    largest = counts[1:].argmax() + 1
    total_pixels = activity_map.size
    field = activity_map  # all surviving clusters

    if nb > max_clusters :
        if verbose :
            print("too many -- not place like")
        return False, field

    if counts[largest] > max_field_fraction * total_pixels :
        if verbose :
            print("too large, not place like :", np.count_nonzero(field) / total_pixels)
        return False, field

    return True, field

def is_empty(activity_map) :
    """ Stub for deciding whether a spatial activity map is completely empty."""
    is_empty = False
    if np.all(activity_map == 0) :
        is_empty = True
    return is_empty

def empty_fields(fields) :
    """ Returns all unit ids that show no spatial activity in all head directions,
    and in any of the head directions.
    """
    fields = np.squeeze(fields)
    n_units = fields.shape[2]
    empty = np.zeros((6,n_units))
    for f,i in zip(np.rollaxis(fields,2),range(n_units)) :
        for hd,j in zip(f,range(6)) :
            hd = hd.reshape(25,25)
            empty[j,i] = is_empty(hd)
    a = np.all(empty, axis=0)        
    ids_all = np.atleast_1d(np.squeeze(np.array(np.where(a))))
    b = np.any(empty, axis=0)
    ids_any = np.atleast_1d(np.squeeze(np.array(np.where(b))))
    ids_any = np.setdiff1d(ids_any, ids_all)
    return ids_all, ids_any

def center_distance_threshold(max_center_distance, relative, map_height) :
    """ Convert a center-distance threshold to pixels.

    If relative is True, max_center_distance is interpreted as a fraction of
    the map height; otherwise it is an absolute number of pixels.
    """
    if relative :
        return max_center_distance * map_height
    return max_center_distance


def calculate_pdist(fields_sliced, map_shape=None) :
    """ Pairwise euclidean distances (in pixels) between the centers of
    each unit's per HD fields.

    Keyword arguments :
    fields_sliced : array (n_units, n_directions, n_pixels) of flattened maps
    map_shape     : (rows, cols) of each map; inferred as square if None

    Returns a list holding one condensed pairwise-distance vector per unit.
    """
    fields_sliced = np.asarray(fields_sliced)
    n_units, n_dirs, n_pix = fields_sliced.shape
    if map_shape is None :
        side = int(round(np.sqrt(n_pix)))
        map_shape = (side, side)
    n_cols = map_shape[1]
    centers = np.zeros((n_units, n_dirs, 2))
    for i in range(n_units) :
        for j in range(n_dirs) :
            centers[i, j, :] = divmod(np.argmax(fields_sliced[i, j]), n_cols)
    return [pdist(centers[i]) for i in range(n_units)]

def plot_field(fields, show_centers=True) :

    fig,ax = plt.subplots(fields.shape[0],fields.shape[1],
                          figsize=(fields.shape[1],fields.shape[0]))
    centers = np.zeros((len(fields),6,2))
    for f,i in zip(fields,range(len(fields))) :
        for angle,j in zip(f,range(len(f))) :
            centers[i,j,:] = divmod((np.argmax(angle)),25)
            angle = angle.reshape(25,25)

            ax[i,j].imshow(nd.gaussian_filter(angle,sigma=2),cmap='jet',
              origin='lower')
            if show_centers :
                ax[i,j].scatter([centers[i,j,1]],[centers[i,j,0]],c='white',
                  s=50)
            ax[i,j].axis('off')

def _rotate_hd_fields(fields, n_hd, width, height, start_angle=90.0) :
    """ Rotate each head-direction field so egocentric fields align.

    Used for vector-cell detection.
    """
    step = 360.0 / n_hd
    rotated = np.zeros_like(fields)
    n_units = fields.shape[2]
    for i in range(n_units) :
        for j in range(n_hd) :
            angle = (start_angle - step * j) % 360
            hd = fields[j, :, i].reshape(height, width)
            hd = nd.rotate(hd, start_angle - angle, reshape=False, cval=np.min(hd))
            rotated[j, :, i] = hd.flatten()
    return rotated


def _classify(fields, keep, use_mean=True, preprocess=None,
              max_center_distance=5, relative=False,
              n_hd=6, width=25, height=25) :
    """ Shared pipeline for place_like / vector_cells / modulated.

    Keyword arguments :
    keep       : "within" -> keep a unit when ALL pairwise centre distances are
                 <= threshold; "beyond" -> keep when ANY distance is > threshold
    use_mean   : also require the across-direction mean map to be a field
    preprocess : optional fn(fields, n_hd, width, height) -> fields, applied
                 before classification (used to rotate vector-cell fields)
    n_hd, width, height : number of head directions and map dimensions

    Returns (f, ids, pd) : largest-cluster maps, unit ids, and per-unit pairwise
    centre distances (in pixels) for the units that pass.
    """
    n_pix = width * height
    fields = np.squeeze(fields)
    n_units = fields.shape[2]

    if preprocess is not None :
        fields = preprocess(fields, n_hd, width, height)

    if use_mean :
        mean_fields = np.mean(fields, axis=0).reshape((1, n_pix, n_units))
        stacked = np.vstack((fields, mean_fields))
    else :
        stacked = fields
    n_maps = stacked.shape[0]

    is_field = np.zeros((n_maps, n_units))
    largest_field = np.zeros((n_maps, n_pix, n_units))
    for i in range(n_units) :
        for j in range(n_maps) :
            is_, f_ = is_blob(stacked[j, :, i].reshape(height, width))
            is_field[j, i] = is_
            largest_field[j, :, i] = f_.flatten()

    thr = center_distance_threshold(max_center_distance, relative, height)

    ids = np.atleast_1d(np.flatnonzero(np.all(is_field, axis=0)))
    if ids.size == 0 :
        return np.empty((0, n_maps, n_pix)), ids, []

    f = largest_field[:, :, ids].transpose(2, 0, 1)      # (n_ids, n_maps, n_pix)
    pd = calculate_pdist(f[:, :n_hd, :], map_shape=(height, width))

    pd_arr = np.array(pd)
    if keep == "within" :
        survive = np.all(pd_arr <= thr, axis=1)
    else :  # "beyond"
        survive = np.any(pd_arr > thr, axis=1)

    ids = ids[survive]
    f = f[survive]
    pd = [p for p, s in zip(pd, survive) if s]
    return f, ids, pd


def place_like(fields, max_center_distance=5, relative=False,
               n_hd=6, width=25, height=25) :
    """ Units whose field centre is stable across all head directions
    (and whose across-direction mean is also a field). """
    return _classify(fields, keep="within", use_mean=True, preprocess=None,
                     max_center_distance=max_center_distance, relative=relative,
                     n_hd=n_hd, width=width, height=height)

def vector_cells(fields, max_center_distance=5, relative=False,
                 n_hd=6, width=25, height=25) :
    """ Units whose fields become stable once each head-direction field is
    rotated to a common reference (egocentric / vector-like). """
    return _classify(fields, keep="within", use_mean=True,
                     preprocess=_rotate_hd_fields,
                     max_center_distance=max_center_distance, relative=relative,
                     n_hd=n_hd, width=width, height=height)

def modulated(fields, max_center_distance=5, relative=False,
              n_hd=6, width=25, height=25) :
    """ Units that are a field in every head direction but whose centre shifts
    by more than the threshold in at least one of them. """
    return _classify(fields, keep="beyond", use_mean=False, preprocess=None,
                     max_center_distance=max_center_distance, relative=relative,
                     n_hd=n_hd, width=width, height=height)