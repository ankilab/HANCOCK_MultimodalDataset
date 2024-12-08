import qupath.lib.io.TMAScoreImporter

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
def hierarchy = getCurrentHierarchy()
def importer = new TMAScoreImporter()
def return_val = importer.importFromCSV(tma_map_path, hierarchy)
if(return_val != 72){
   print 'Unable to import all case IDs! Only imported ' + return_val + ' out of 72\n'
}