// Check if grid is 6 columns x 12 rows

def tma_name = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())

def hierarchy = getCurrentHierarchy()
grid = hierarchy.getTMAGrid()

cols = grid.getGridWidth()
rows = grid.getGridHeight()

if (cols != 6 || rows != 12)
{
    print tma_name + ": " + cols + " x " + rows + "\n"
}