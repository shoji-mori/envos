import numpy as np
from scipy import interpolate
import itertools

## For any data
def calc_datacor(
    data1,
    data2,
    mode="cube",
    method="ZNCC",
    threshold1=0,
    threshold2=0,
    floor=0,
    ranges=[],
    axes=[],
    norm1=False,
    norm2=False,
    interp_option={},
    get_images=False,
    preprocess_func=None,
):
    ## Set axes
    axes_data = data1.get_axes()
    axes_newgrid = []
    for _ax_data, _range, _ax_user in itertools.zip_longest(axes_data, ranges, axes):
        # C-56 fix: when _range is None (ranges shorter than axes), use full axis range
        if _range is None:
            newax = _ax_data if _ax_user is None else _ax_user
        else:
            cond = (_range[0] < _ax_data) & (_ax_data < _range[1])
            newax = _ax_data[cond] if _ax_user is None else _ax_user
        axes_newgrid.append(newax)

    grid = np.meshgrid(*axes_newgrid, indexing="ij")
    newgrid = np.stack(grid, axis=-1)
    _interp_option = {}  # {"bounds_error":False, "fill_value":0}
    _interp_option.update(interp_option)

    ## Make data
    im1 = interpolate.interpn(
        data1.get_axes(), data1.get_I(), newgrid, **_interp_option
    )
    im2 = interpolate.interpn(
        data2.get_axes(), data2.get_I(), newgrid, **_interp_option
    )

    ## Preprocess
    # C-56 fix: compute both factors before applying them (symmetric preprocessing)
    if preprocess_func is not None:
        fac1 = preprocess_func(im1, im2, axes_newgrid)
        fac2 = preprocess_func(im2, im1, axes_newgrid)
        im1 *= fac1
        im2 *= fac2

    ## Normalization
    if norm1:
        im1 /= np.max(im1)

    if norm2:
        im2 /= np.max(im2)

    ## Threshold
    if threshold1 is not None:
        im1 = np.where(im1 > threshold1, im1, floor)

    if threshold2 is not None:
        im2 = np.where(im2 > threshold2, im2, floor)

    ## Check data shape
    if im1.shape != im2.shape:
        raise Exception("Wrong data shape")

    ## Calculate data correlation
    if method == "ZNCC":
        res = calc_ZNCC_data(im1, im2)
    elif method == "SSD":
        res = calc_SSD_data(im1, im2)

    if np.isnan(res):
        raise Exception("Nan!")

    if get_images:
        d1 = data1.copy()
        d1.set_I(im1)
        d1.set_axes(axes_newgrid)
        d2 = data2.copy()
        d2.set_I(im2)
        d2.set_axes(axes_newgrid)
        return res, (d1, d2)
    else:
        return res


def calc_ZNCC_data(im1, im2):
    indmax1 = np.array(im1.shape)
    Vpix = np.prod(indmax1)
    sumAB = np.sum(im1 * im2)
    sumA = np.sum(im1)
    sumB = np.sum(im2)
    sumAsq = np.sum(im1**2)
    sumBsq = np.sum(im2**2)
    ZNCC_AB = (Vpix * sumAB - sumA * sumB) / np.sqrt(
        (Vpix * sumAsq - sumA**2) * (Vpix * sumBsq - sumB**2)
    )
    return ZNCC_AB


def calc_SSD_data(im1, im2):
    return np.sum((im1 - im2) ** 2)
