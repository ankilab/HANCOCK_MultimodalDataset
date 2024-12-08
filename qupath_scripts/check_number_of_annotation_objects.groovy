// Get slide name
def imageServer = getCurrentImageData().getServer()
def name = GeneralTools.getNameWithoutExtension(imageServer.getMetadata().getName())

annotations = getAnnotationObjects().findAll()
def s = annotations.size()
if (annotations.size() != 1) {
    s += " -> slide: " + name
}
print s