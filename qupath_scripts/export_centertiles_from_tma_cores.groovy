import javax.imageio.ImageIO
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.interfaces.ROI


// Define output resolution (width of one pixel in micrometers)
double requestedPixelSize = 1

// Define tile size [micrometers]
int tileSizeMu = 1024

// Create output directory inside the project
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'tiles')
mkdirs(dirOutput)

// Get QuPath data
def imageData = getCurrentImageData()
def server = getCurrentServer()
def path = server.getPath()

// Calculate downsample factor
double pixelSize = server.getPixelCalibration().getAveragedPixelSize() // 0.194475 or 0.2634
double downsample = requestedPixelSize / pixelSize
def tileSize = (int)Math.round(tileSizeMu / pixelSize)

// Get TMA cores
def hierarchy = getCurrentHierarchy()
def cores = hierarchy.getTMAGrid().getTMACoreList().findAll()

// Get image plane
imagePlane = getCurrentViewer().getImagePlane()

def caseID = ""

for (core in cores) {
    // Get case ID (patient ID)
    previousCaseID = caseID
    caseID = core.getCaseID()
    
    // Tile list
    def rois = []
    
    if(caseID != null) {
        if(caseID == previousCaseID) {
           coreNum = 0
        }
        else {
           coreNum = 1 
        }
        
        // Get centroid coordinates [px]
        roi = core.getROI()
        x = roi.getCentroidX() 
        y = roi.getCentroidY()
        
        print tileSize/2
        
        // Create ROI (needs coordinates of top left corner)
        int x = (int)(x - tileSize / 2)
        int y = (int)(y - tileSize / 2)
        print x
        print y
        tileRoi = ROIs.createRectangleROI(x, y, tileSize, tileSize, imagePlane)
        
        
        //// Optionally create annotation objects from ROIs
        //def annotation = PathObjects.createAnnotationObject(tileRoi)
        //addObject(annotation)
        
        // Save tile to .png
        tile = server.readRegion(RegionRequest.createInstance(path, downsample, tileRoi))
        tile_path = caseID + "_core" + coreNum + "_tile.png"
        ImageIO.write(tile, "png", new File(dirOutput, tile_path))
        
    }
}