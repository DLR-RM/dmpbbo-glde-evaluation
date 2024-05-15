""" Module for geting a color pallette. """
def get_colorpalette():
    """
    Get a color pallette as a dict of dict of hex colors.

    These are the DLR corporate colors.

    @return: A color pallette as a dict of dict of hex colors.
    """
    cp = {
        "prim": ["#000000", "#666666", "#b9cad2", "#ffffff"],
        "blue": ["#00658b", "#3b98cb", "#6cb9dc", "#a7d3ec", "#d1e8fa"],
        "yellow": ["#d2ae3d", "#f2cd51", "#f8de53", "#fcea7a", "#fff8be"],
        "green": ["#82a043", "#a6bf51", "#cad55c", "#d9df78", "#e6eaaf"],
        "gray": ["#666666", "#868585", "#b1b1b1", "#cfcfcf", "#ebebeb"],
    }
    return cp
