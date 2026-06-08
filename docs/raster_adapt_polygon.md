# Raster Adapt Polygon Pipeline

Pipeline name:

```yaml
pipeline: "raster_adapt_polygon"
```

This pipeline adapts an existing raster inside a user-defined polygon. It keeps the
original raster outside the polygon and reconstructs the topography inside the
polygon using reference elevations taken from the surrounding area.

The first implementation focuses on boundary-based interpolation, so that the
modified area respects the local terrain around the polygon boundary.

---

## 1. What problem this pipeline solves

Use this pipeline when you have a