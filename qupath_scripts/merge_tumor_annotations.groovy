// Get slide name
def imageServer = getCurrentImageData().getServer()
def name = GeneralTools.getNameWithoutExtension(imageServer.getMetadata().getName())

// Find all annotations
annotations = getAnnotationObjects().findAll()

// Set class to "Tumor" in case it is not set yet
def tumor = getPathClass("Tumor")
for (ann in annotations) {
    ann.setPathClass(tumor)
}

// Merge to a single annotation object per slide
mergeAnnotations(annotations)