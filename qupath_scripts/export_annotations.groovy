// Create output directory inside the project
def dirOutput = buildFilePath(PROJECT_BASE_DIR, "annotations")
mkdirs(dirOutput)

// Define output path
def slide_name = GeneralTools.getNameWithoutExtension(getProjectEntry().getImageName())
String file_path = buildFilePath(dirOutput, slide_name  + ".geojson")

// Get annotations
def annotations = getAnnotationObjects()

// 'FEATURE_COLLECTION' is standard GeoJSON format for multiple objects
exportObjectsToGeoJson(annotations, file_path, "FEATURE_COLLECTION")

print "Exported " + name + ".geojson"