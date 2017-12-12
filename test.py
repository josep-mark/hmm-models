from homework11 import *
import unittest

class HMMTest(unittest.TestCase):

	def test_load_corpus(self):
		s = load_corpus("Brown_sample.txt")
		self.assertEquals(len(s), 2824)

	def test_load_corpus2(self):
		s = load_corpus("Brown_sample.txt")
		self.assertEquals(s[1208:1228], "to the best interest")

	def test_load_parameters(self):
		p = load_parameters("homework11_simple.pickle")
		self.assertEquals(p[1][1], {1: -0.6931471805599453, 2: -0.6931471805599453})

	def test_load_parameters2(self):
		p = load_parameters("homework11_prob_vector.pickle")
		self.assertEquals(p[1][1], {1: -0.9394332049935226,
									2: -1.663989250166688,
									3: -2.563281364593492,
									4: -1.070849586518945})

	def test_get_parameters(self):
		p = load_parameters("homework11_simple.pickle")
		h = HMM (p)
		n = h.get_parameters()
		self.assertEquals(n[1][1], {1: 0.5 , 2: 0.5})

	def test_get_parameters2(self):
		p = load_parameters("homework11_prob_vector.pickle")
		h = HMM (p)
		n = h.get_parameters()
		self.assertEquals(n[1][1], {1: 0.3908493040227309,
									2: 0.18938197908059773,
									3: 0.0770514911339938,
									4: 0.3427172257626776})

	def test_forward(self):
		s = "the cat ate the rat"
		p = load_parameters ("homework11_simple.pickle")
		h = HMM(p)
		f = h.forward(s)
		print f[10]
		# self.assertEquals(f[10], {1: -22.0981588201684 ,
		# 						2: -22.0981588201684})

	def test_forward2(self):
		s = load_corpus("Brown_sample.txt")
		p = load_parameters("homework11_prob_vector.pickle")
		h = HMM(p)
		f = h.forward(s)
		print f[1400]
		# self.assertEquals(f[1400], {1: -4570.10024680558,
		# 							2: -4569.896509256886,
		# 							3: -4569.956231590213,
		# 							4: -4569.542222483007})




if __name__ == '__main__':
    unittest.main()