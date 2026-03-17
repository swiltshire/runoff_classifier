# src/utils/make_vrt.py
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
import rasterio

def write_mosaic_vrt(vrt_path: str, tif_paths: list[str]) -> None:
    """
    write a simple mosaic VRT from a list of geotiffs without osgeo.gdal.

    assumptions:
      - all tiles are north-up (no rotation/skew)
      - all tiles share the same crs and pixel size
      - we mirror band count + dtypes from the first file
      - relative paths are stored inside the VRT for portability
    """
    if not tif_paths:
        raise ValueError("no input tiles provided")

    # map numpy dtype -> gdal vrt dataType
    def gdal_dtype_of(np_dtype: str) -> str:
        np_dtype = str(np_dtype).lower()
        mapping = {
            "uint8":   "Byte",
            "int8":    "Byte",     # gdal vrt has no Int8; Byte is the 8-bit type
            "uint16":  "UInt16",
            "int16":   "Int16",
            "uint32":  "UInt32",
            "int32":   "Int32",
            "float32": "Float32",
            "float64": "Float64",
        }
        if np_dtype not in mapping:
            raise ValueError(f"unsupported dtype for VRT: {np_dtype}")
        return mapping[np_dtype]

    # open first tile for template
    with rasterio.open(tif_paths[0]) as src0:
        crs_wkt = src0.crs.to_wkt() if src0.crs else ""
        band_count = src0.count
        dtypes0 = list(src0.dtypes)
        t0 = src0.transform
        px_w = t0.a
        px_h = -t0.e if t0.e < 0 else t0.e  # assume north-up

    # global bounds across tiles
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    transforms = {}
    sizes = {}
    for p in tif_paths:
        with rasterio.open(p) as s:
            transforms[p] = s.transform
            sizes[p] = (s.width, s.height)
            left, bottom, right, top = s.bounds
            if left   < minx: minx = left
            if bottom < miny: miny = bottom
            if right  > maxx: maxx = right
            if top    > maxy: maxy = top

    # global raster size in pixels
    width  = int(round((maxx - minx) / px_w))
    height = int(round((maxy - miny) / px_h))

    # root
    os.makedirs(os.path.dirname(vrt_path) or ".", exist_ok=True)
    vrt = ET.Element("VRTDataset", attrib={"rasterXSize": str(width), "rasterYSize": str(height)})

    # srs
    if crs_wkt:
        srs = ET.SubElement(vrt, "SRS")
        srs.text = crs_wkt

    # geotransform: origin at (minx, maxy), pixel size (px_w, -px_h)
    geot = ET.SubElement(vrt, "GeoTransform")
    geot.text = f"{minx}, {px_w}, 0.0, {maxy}, 0.0, {-px_h}"

    # nodata from the first file (optional)
    with rasterio.open(tif_paths[0]) as src0_chk:
        nodata_val = src0_chk.nodata

    # per-band nodes with correct GDAL dataType
    band_nodes = []
    for b in range(1, band_count + 1):
        gdal_dt = gdal_dtype_of(dtypes0[b - 1])
        band = ET.SubElement(vrt, "VRTRasterBand", attrib={"dataType": gdal_dt, "band": str(b)})
        if nodata_val is not None:
            nd = ET.SubElement(band, "NoDataValue")
            nd.text = str(nodata_val)
        band_nodes.append(band)

    # add each tif as a SimpleSource for each band (relative paths)
    vrt_dir = os.path.dirname(os.path.abspath(vrt_path))
    for p in tif_paths:
        t = transforms[p]
        w, h = sizes[p]
        x0 = t.c
        y0 = t.f
        x_off = int(round((x0 - minx) / px_w))
        y_off = int(round((maxy - y0) / px_h))
        rel = os.path.relpath(os.path.abspath(p), vrt_dir)

        for b in range(band_count):
            band = band_nodes[b]
            ss = ET.SubElement(band, "SimpleSource")

            sf = ET.SubElement(ss, "SourceFilename", attrib={"relativeToVRT": "1"})
            sf.text = rel

            sb = ET.SubElement(ss, "SourceBand")
            sb.text = str(b + 1)

            ET.SubElement(ss, "SrcRect", attrib={"xOff": "0", "yOff": "0", "xSize": str(w), "ySize": str(h)})
            ET.SubElement(ss, "DstRect", attrib={"xOff": str(x_off), "yOff": str(y_off), "xSize": str(w), "ySize": str(h)})

    # write xml
    ET.ElementTree(vrt).write(vrt_path, encoding="UTF-8", xml_declaration=True)