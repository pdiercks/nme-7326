import gmsh


def create_lpanel_grid(xmin, xmax, ymin, ymax, z=0., lc=1., num_cells=None, recombine=False, facets=False, out_file=None, gmsh_options=None):
    """create partition of the domain 
    Ω = [xmin, xmax] x [ymin, ymax] \\ 
        [xmin + (xmax-xmin)/2, xmax] x [ymin, ymin + (ymax-ymin)/2]
    (quadrilateral domain with bottom right quadrant missing)

                              x0 = xmin + (xmax-xmin)/2
                              y0 = ymin + (ymax-ymin)/2
    4--------3--------2       p0 = (x0, y0) 
    |                 |       p1 = (XMAX, y0)
    |                 |       p2 = (XMAX, YMAX)
    |                 |       p3 = (x0, YMAX)
    5        0--------1       p4 = (XMIN, YMAX)
    |        |                p5 = (XMIN, y0)  
    |        |                p6 = (XMIN, YMIN)
    |        |                p7 = (x0, YMIN)
    6--------7

    Parameters
    ----------
    xmin : float
    xmax : float
    ymin : float
    ymax : float
    z : optional, float
    lc : optional, float
        Float to control mesh size in Gmsh.
    num_cells : optional, int
        If not None, the number of cells in x and y.
        Results in n**2 - (n/2)**2 cells if recombine is True.
    recombine : optional, bool
        If True, recombine triangles into quadrilaterals.
    facets : optional, bool
        Create physical groups for boundaries.
    out_file : optioanl, str
        FilePath to write the grid.
    gmsh_options : optional, dict
        Gmsh options. Will be set via gmsh.option.setNumber.
    """

    gmsh.initialize()
    gmsh.model.add("lpanel")

    if gmsh_options is not None:
        try:
            for key, value in gmsh_options.items():
                gmsh.option.setNumber(key, value)
        except KeyError as err:
            raise err("Unknown Gmsh Option ...")


    # COOS
    ox = xmin + (xmax-xmin)/2
    oy = ymin + (ymax-ymin)/2

    p0 = gmsh.model.geo.addPoint(xmin, ymin, z, lc)
    p1 = gmsh.model.geo.addPoint(ox, ymin, z, lc)
    p2 = gmsh.model.geo.addPoint(ox, oy, z, lc)
    p3 = gmsh.model.geo.addPoint(xmax, oy, z, lc)
    p4 = gmsh.model.geo.addPoint(xmax, ymax, z, lc)
    p5 = gmsh.model.geo.addPoint(ox, ymax, z, lc)
    p6 = gmsh.model.geo.addPoint(xmin, ymax, z, lc)
    p7 = gmsh.model.geo.addPoint(xmin, oy, z, lc)

    # boundary
    l0 = gmsh.model.geo.addLine(p0, p1)
    l1 = gmsh.model.geo.addLine(p1, p2)
    l2 = gmsh.model.geo.addLine(p2, p3)
    l3 = gmsh.model.geo.addLine(p3, p4)
    l4 = gmsh.model.geo.addLine(p4, p5)
    l5 = gmsh.model.geo.addLine(p5, p6)
    l6 = gmsh.model.geo.addLine(p6, p7)
    l7 = gmsh.model.geo.addLine(p7, p0)

    # inner lines
    l52 = gmsh.model.geo.addLine(p5, p2)
    l72 = gmsh.model.geo.addLine(p7, p2)

    # line loops
    loop_1 = gmsh.model.geo.addCurveLoop([l0, l1, -l72, l7])
    loop_2 = gmsh.model.geo.addCurveLoop([l72, -l52, l5, l6])
    loop_3 = gmsh.model.geo.addCurveLoop([l2, l3, l4, l52])

    # surfaces
    quadrant_1 = gmsh.model.geo.addPlaneSurface([loop_1])
    quadrant_2 = gmsh.model.geo.addPlaneSurface([loop_2])
    quadrant_3 = gmsh.model.geo.addPlaneSurface([loop_3])

    if num_cells is not None:
        if num_cells % 2:
            raise ValueError(f"'num_cells' needs to be an even number, but is {num_cells=}")
        n = int(num_cells / 2)

        gmsh.model.geo.mesh.setTransfiniteCurve(l0, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l1, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l2, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l3, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l4, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l5, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l6, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l7, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l52, n + 1)
        gmsh.model.geo.mesh.setTransfiniteCurve(l72, n + 1)

        gmsh.model.geo.mesh.setTransfiniteSurface(quadrant_1, "Left")
        gmsh.model.geo.mesh.setTransfiniteSurface(quadrant_2, "Left")
        gmsh.model.geo.mesh.setTransfiniteSurface(quadrant_3, "Left")

        if recombine:
            # setRecombine(dim, tag, angle=45.0)
            gmsh.model.geo.mesh.setRecombine(2, quadrant_1)
            gmsh.model.geo.mesh.setRecombine(2, quadrant_2)
            gmsh.model.geo.mesh.setRecombine(2, quadrant_3)

    gmsh.model.geo.synchronize()
    gmsh.model.addPhysicalGroup(2, [quadrant_1, quadrant_2, quadrant_3])

    if facets:
        gmsh.model.add_physical_group(1, [l2], 11, name="y0")
        gmsh.model.add_physical_group(1, [l3], 12, name="right")
        gmsh.model.add_physical_group(1, [l4, l5], 13, name="top")
        gmsh.model.add_physical_group(1, [l6, l7], 14, name="left")
        gmsh.model.add_physical_group(1, [l0], 15, name="bottom")
        gmsh.model.add_physical_group(1, [l1], 16, name="x0")

    filepath = out_file or "./lpanel.msh"

    gmsh.model.mesh.generate(2)
    gmsh.write(filepath)
    gmsh.finalize()
