from cStringIO import StringIO
import logging
import numpy as np
import os
import random
import requests
import svgwrite
import urllib

from scipy.misc import imresize
from skimage.draw import line_aa

QUICK_DRAW_BASE_URL = 'https://storage.googleapis.com/quickdraw_dataset/sketchrnn'
QUICK_DRAW_CATEGORIES_URL = 'https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt'
TESTING_DRAW_CATEGORIES_URL = 'categories.txt'

def get_bounds(strokes):
	"""Given a 3-stroke sequence returns the bounds for the respective sketch

	Args:
		strokes: A 3-stroke sequence representing a single sketch

	Returns:
		(min_x, max_x, min_y, max_y): bounds of the sketch
	"""
	min_x, max_x, min_y, max_y = (0, 0, 0, 0)
	abs_x, abs_y = (0, 0)

	for i in range(strokes.shape[0]):
		dx, dy = strokes[i, :2]
		abs_x += dx
		abs_y += dy
		min_x = min(min_x, abs_x)
		max_x = max(max_x, abs_x)
		min_y = min(min_y, abs_y)
		max_y = max(max_y, abs_y)

	return (min_x, max_x, min_y, max_y)

def strokes_to_npy(strokes):
	"""Given a 3-stroke sequence returns the sketch in a numpy array

	Args:
		strokes: A 3-stroke sequence representing a single sketch

	Returns:
		img: A grayscale 2D numpy array representation of the sketch
	"""
	min_x, max_x, min_y, max_y = get_bounds(strokes)
	dims = (50 + max_x - min_x, 50 + max_y - min_y)
	img = np.zeros(dims, dtype=np.uint8)
	abs_x = 25 - min_x
	abs_y = 25 - min_y
	pen_up = 1
	for i in range(strokes.shape[0]):
		dx, dy = strokes[i, :2]

		if pen_up == 0:
			rr, cc, val = line_aa(abs_x, abs_y, abs_x + dx, abs_y + dy)
			img[rr, cc] = val * 255

		abs_x += dx
		abs_y += dy
		pen_up = strokes[i, 2]

	# TODO: Why do we need to transpose? Fix get_bounds accordingly
	return img.T

def reshape_to_square(img, size):
	"""Given any size numpy array return

	Args:
		img: A grayscale 2D numpy array representation of the sketch

	Returns:
		img_sq: A grayscale 2D numpy array representation of the sketch fitted
				in a size x size square.
	"""
	# TODO: make sure this formula is correct
	# TODO: draw in a square instead of rescaling
	img_resize = imresize(img, float(size) / max(img.shape))
	w_, h_ = img_resize.shape
	x, y = ((size - w_) / 2, (size - h_) / 2)

	img_sq = np.zeros((size, size), dtype=np.uint8)
	img_sq[x:x + w_, y:y + h_] = img_resize

	return img_sq

class DataLoader(object):
	"""Class for loading data."""

	def __init__(self,
				dataset_type='train',
				batch_size=20,
				img_size=225,
				count_per_category=2000,
				data_format='bitmap',
				datastore_dir='/tmp/quick-draw-net/data/',
				dataset_base_url=QUICK_DRAW_BASE_URL,
				categories_file=TESTING_DRAW_CATEGORIES_URL,
				shuffle=False):

		self.dataset_type = dataset_type
		self.batch_size = batch_size
		self.img_size = img_size
		self.count_per_category = count_per_category
		self.data_format = data_format
		self.datastore_dir = datastore_dir
		self.dataset_base_url = dataset_base_url
		
		if not os.path.exists(datastore_dir):
			logging.info('Created Datastore Directory %s' % datastore_dir)
			os.makedirs(datastore_dir)

		self.categories_path = {}
		self.categories_url = {}
		self.idx_to_categories = self.load_categories(categories_file)

		assert batch_size <= count_per_category * len(self.idx_to_categories)
		assert (count_per_category * len(self.idx_to_categories)) % batch_size == 0

		self.idx_category = 0
		self.idx = 0

		self.strokes = {}
		category = self.idx_to_categories[self.idx_category]
		if category not in self.strokes:
			self.load_strokes(category)
	
		# TODO: Store bitmaps so that we don't have to recompute bitmaps
		# if self.data_format == 'bitmap':
		# 	self.bitmaps = {}

	def load_categories(self, categories_file):
		"""Loads the categories line seperated file with all the categories to
		loaded by the DataLoader. Adds the path to the dataset file in the
		categories_path dictionary.

		Args:
			categories_file: path to the line seperated file with desired category names
		"""
		categories = []
		if categories_file.startswith('http://') or categories_file.startswith('https://'):
			logging.info('Downloading categories file from %s' % categories_file)
			response = requests.get(categories_file)
			if response.status_code != 200:
				raise IOError(msg='Request to %s responded with status %d'
						% (categories_file, response.status_code))
			categories = response.content.split('\n')
		else:
			logging.info('Opening categories file at %s' % categories_file)
			categories = open(categories_file).read().split('\n')

		# filter skiped lines and categories that are skipped (starts with #)
		categories = list(filter(lambda x: len(x) > 0 and x[0] != '#', categories))
		# add the urls if they are valid and give status 200

		for category in categories:
			# Smaller Dataset
			category_path = os.path.join(self.datastore_dir, '%s.npz' % category)
			if os.path.isfile(category_path):
				self.categories_path[category] = category_path

			category_url = os.path.join(self.dataset_base_url, '%s.npz' % category)
			logging.info('Pinging to check if exists %s' % category_url)

			response = requests.head(category_url)
			if response.status_code == 200:
				self.categories_url[category] = category_url
				logging.info('Success! Found %s' % category_url)
			else:
				logging.warn('Category %s was not found' % category)

			# Full Dataset
			category_full_path = os.path.join(self.datastore_dir, '%s.full.npz' % category)
			if os.path.isfile(category_full_path):
				self.categories_path['%s.full' % category] = category_full_path

			category_full_url = os.path.join(self.dataset_base_url, '%s.full.npz' % category)
			logging.info('Pinging to check if exists %s' % category_full_url)

			response = requests.head(category_full_url)
			if response.status_code == 200:
				self.categories_url['%s.full' % category] = category_full_url
				logging.info('Success! Found %s' % category)
			else:
				logging.warn('Category %s (full) was not found' % category)

		return categories

	def load_strokes(self, category, full=False):
		"""Loads the strokes for the requested category. It Downloads the 
		dataset to the datastore_dir if not already available.

		Args:
			category: category of the dataset to be loaded
			full: a boolean flag to download the complete dataset file for
					the given category
		"""
		category_ = category if not full else '%s.full' % category

		if category_ not in self.categories_path and category_ not in self.categories_url:
			raise ValueError('Could not download category %s because if does not exist' % category_)

		# download the data if don't have a local copy
		if category_ not in self.categories_path:
			logging.info('Downloading category %s to %s' % (category_, self.datastore_dir))
			response = requests.get(self.categories_url[category_])
			if response.status_code != 200:
				raise IOError(msg='Request to %s responded with status %d'
						% (self.categories_url[category_], response.status_code))

			with open(os.path.join(self.datastore_dir, '%s.npz' % category_), 'w') as handle:
				# TODO: log downloading progress
				for block in response.iter_content(1024):
					handle.write(block)

			self.categories_path[category_] = os.path.join(self.datastore_dir, '%s.npz' % category_)

		# load the category data into memory
		logging.info('Loading %s data to memory' % category_)
		strokes = np.load(StringIO(open(self.categories_path[category_]).read()))

		# TODO: look for optimization, loads all then returns count_per_category items
		self.strokes[category_] = strokes[self.dataset_type][:self.count_per_category]
		logging.info('Loaded %s data' % category_)

	def _increment_idx(self, step_size):
		self.idx += step_size
		if self.idx >= self.count_per_category:
			self.idx_category += 1
			self.idx_category %= len(self.idx_to_categories)
			category = self.idx_to_categories[self.idx_category]
			if category not in self.strokes:
				self.load_strokes(category)
			self.idx %= self.count_per_category

	def next_batch(self):
		category = self.idx_to_categories[self.idx_category]
		idx_end = min(self.idx+self.batch_size, self.count_per_category)
		strokes_batch = self.strokes[category][self.idx:idx_end]
		labels_batch = np.array([self.idx_category] * (idx_end - self.idx))

		self._increment_idx(self.batch_size)
		
		if len(strokes_batch) < self.count_per_category:
			category = self.idx_to_categories[self.idx_category]
			np.append(strokes_batch, self.strokes[category][0:self.idx])
			np.append(labels_batch, np.array([self.idx_category] * self.idx))

		if self.data_format != 'bitmap':
			return strokes_batch

		# convert to strokes to bitmap and return
		strokes_to_sqnp = lambda x: reshape_to_square(strokes_to_npy(x), self.img_size)
		bitmaps_batch = np.array(map(strokes_to_sqnp, strokes_batch))
		# TODO: Store bitmaps so that we don't have to recompute bitmaps
		return np.expand_dims(bitmaps_batch, axis=3), labels_batch
		