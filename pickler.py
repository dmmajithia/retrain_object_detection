import pickle

class SubImageData():
	def __init__(self, path, ID = 0):
		self.path = path
		self.names = set()
		self.boxes = {}
		self.id = ID
		self.it = 0
	def add(self,filename,box):
		self.names.add(filename)
		self.boxes[filename] = box
	def remove(self,filename):
		if filename in self.names:
			self.names.remove(filename)
			del self.boxes[filename]

class ImageData():
	def __init__(self, path):
		self.subs = []
		self.subs.append(SubImageData(path=path))
		self.current_sub = 0

	@staticmethod
	def load(pickle_name):
		with open(pickle_name, 'rb') as f:
			return pickle.load(f)
	def new(self,path):
		sub = [s for s in self.subs if s.path == path]
		if len(sub) > 0:
			self.current_sub = sub[0].id
		else:
			self.subs.append(SubImageData(path=path), len(self.subs))
			self.current_sub = len(self.subs)-1
	def add(self,filename,box):
		self.subs[self.current_sub].add(filename,box)
	def remove(self,filename):
		self.subs[self.current_sub].remove(filename)
	def save(self,pickle_name):
		with open(pickle_name,'wb') as f:
			pickle.dump(self,f)

	

