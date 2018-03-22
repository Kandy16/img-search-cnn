import os, glob

class ImageListCreator(object):
	def __init__(self):
		pass

	# This takes a directory name and looks for jpg images and creates a text file listing those images location.
	def make_list_image_filenames(self, image_path):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		#filename = os.path.join(dir_path , directory_name)

		filename = image_path

		if os.path.exists(filename):
			os.chdir(filename)
			text_file = open(os.path.join(dir_path , "images.txt"), "w")

			types = ('*.jpeg', '*.jpg' , '*.JPEG' , '*.png') # the tuple of file types
			files_grabbed = []
			for files in types:
				files_grabbed.extend(glob.glob(files))

			for file in files_grabbed:
				text_file.write(os.path.join(filename , file) + "\n")
			text_file.close()
			print("images.txt created at location : " + dir_path)
			os.chdir(dir_path)
		else:
			print("No Folder exist of name \"" +  image_path + "\" Please create and put images into it")


if __name__ == "__main__":
	# How to use
	fm = ImageListCreator()
	#"Parameter: folder name where your images are located"
	fm.make_list_image_filenames("images")