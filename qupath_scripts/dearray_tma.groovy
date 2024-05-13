if (!isTMADearrayed()) {
	runPlugin('qupath.imagej.detect.dearray.TMADearrayerPluginIJ', '{"coreDiameterMM":1.9,"labelsHorizontal":"1-6","labelsVertical":"1-12","labelOrder":"Row first","densityThreshold":5,"boundsScale":105}')
	return;
}
