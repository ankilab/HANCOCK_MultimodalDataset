import qupath.lib.io.TMAScoreImporter


// Tissue detection inside TMA cores (if not executed beforehand)
// ----------------------------------------------------------
//selectTMACores();
//createAnnotationsFromPixelClassifier("tissueDetectionTMACores", 0.0, 0.0)


// Make sure that TMA core labels match the ones in the TMA maps
// columns: 1-6, rows: 1-12. Bottom-right core has label 12-6
// ----------------------------------------------------------
relabelTMAGrid("1-6", "1-12", true)


// Workaround: Match annotation to their parent TMA cores (required after manual changes)
// ----------------------------------------------------------
// Get TMA cores without child objects
def hierarchy = getCurrentHierarchy()
def cores = hierarchy.getTMAGrid().getTMACoreList().findAll {
    it.getChildObjects().size() == 0
}

// Get all annotation objects without parent objects
def annotations = getAnnotationObjects().findAll {
    it.getParent().getName() == null
}
// Match annotations and cores
for (ann in annotations) {
    roi = ann.getROI()
    // Get centroid coordinates of annotation
    x = roi.getCentroidX()
    y = roi.getCentroidY()
    
    for (core in cores) {
        // Get bounding box coordinates
        core_roi = core.getROI()
        xmin = core_roi.getBoundsX()
        ymin = core_roi.getBoundsY()
        xmax = xmin + core_roi.getBoundsWidth()
        ymax = ymin + core_roi.getBoundsHeight()
        
        // Find parent TMA core
        if (x>xmin && x<xmax && y>ymin && y<ymax) {
           print ann
           print core
           print " "
           // Add annotation as child object to the TMA core
           hierarchy.addObjectBelowParent(core, ann, true)
           break
        }
    }
}

// Import TMA maps
// ----------------------------------------------------------
// Open dialog to select directory containing the TMA maps
def tma_map_dir = Dialogs.promptForDirectory(null)
// Alternative: Hard-code the path to the directory for fast execution of 'run for project'
// def tma_map_dir = 'path/to/TMA_Maps'

// Get TMA block number
def tma_name = GeneralTools.getNameWithoutExtension(getCurrentImageData().getServer().getMetadata().getName())
def block_num = (tma_name =~ /block[0-9]{1,2}/)[0]
def tma_map_path = new File(buildFilePath(tma_map_dir.toString(), 'TMA_Map_' + block_num + '.csv'))
print tma_map_path

// import TMA map
def importer = new TMAScoreImporter()
def return_val = importer.importFromCSV(tma_map_path, hierarchy)
if(return_val != 72){
   print 'Unable to import all case IDs! Only imported ' + return_val + ' out of 72\n'
}


// Positive cell detection
// ----------------------------------------------------------
selectAnnotations();
runPlugin('qupath.imagej.detect.cells.PositiveCellDetection', '{"detectionImageBrightfield":"Optical density sum","requestedPixelSizeMicrons":0.5,"backgroundRadiusMicrons":8.0,"backgroundByReconstruction":true,"medianRadiusMicrons":0.0,"sigmaMicrons":1.5,"minAreaMicrons":10.0,"maxAreaMicrons":400.0,"threshold":0.01,"maxBackground":2.0,"watershedPostProcess":true,"excludeDAB":false,"cellExpansionMicrons":5.0,"includeNuclei":true,"smoothBoundaries":true,"makeMeasurements":true,"thresholdCompartment":"Nucleus: DAB OD mean","thresholdPositive1":0.2,"thresholdPositive2":0.4,"thresholdPositive3":0.6000000000000001,"singleThreshold":true}')


// Export TMA measurements to CSV
// ----------------------------------------------------------
// Create output directory inside the project
def dirOutput = buildFilePath(PROJECT_BASE_DIR, 'tma_measurements')
mkdirs(dirOutput)

// Create file name
tma_name = tma_name.replaceAll("\\s", "")
filename = dirOutput + '/' + tma_name + '.csv'
print filename

// Export TMA measurements to .csv
saveTMAMeasurements(filename)