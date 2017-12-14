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
		# print f[10]
		self.assertEquals(f[10], {1: -22.0981588201684 ,
								2: -22.0981588201684})

	def test_forward2(self):
		s = load_corpus("Brown_sample.txt")
		p = load_parameters("homework11_prob_vector.pickle")
		h = HMM(p)
		f = h.forward(s)
		# print f[1400]
		self.assertEquals(f[1400], {1: -4570.10024680558,
									2: -4569.896509256886,
									3: -4569.956231590213,
									4: -4569.542222483007})

	def test_forward_prob1(self):
		s = "the cat ate the rat"
		p = load_parameters("homework11_simple.pickle")
		h = HMM(p)
		self.assertEquals(h.forward_probability(h.forward(s)), -36.972292832050975)

	# def test_forward_prob2(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.forward_probability(h.forward(s)), -9201.34957430782)

	# def test_forward_prob2(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.forward_probability(h.forward(s)), -9201.34957430782)


	def test_backward(self):
		s = "the cat ate the rat"
		p = load_parameters ("homework11_simple.pickle")
		h = HMM(p)
		b = h.backward(s)
		# print b[9]
		self.assertEquals(b[9], {1: -17.513191341497826,
								2: -17.513191341497826})


	def test_backward2(self):
		s = load_corpus("Brown_sample.txt")
		p = load_parameters("homework11_prob_vector.pickle")
		h = HMM(p)
		f = h.backward(s)
		print f[1424]
		self.assertEquals(f[1424], {1: -4553.117090965298,
									2: -4553.249309905892,
									3: -4553.085375790753,
									4: -4553.140279571696})

	# def test_backward_probs(self):
	# 	s = "the cat ate the rat"
	# 	p = load_parameters("homework11_simple.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.backward_probability(h.backward(s), s), -36.97229283205097)

	# def test_backward_probs2(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.backward_probability(h.backward(s), s), -9201.349574307758)

	# def test_xi_matrix(self):
	# 	s = "the cat ate the rat"
	# 	p = load_parameters("homework11_simple.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.xi_matrix(5, s, h.forward(s), h.backward(s))[2], {1: -1.3862943611198943,
	# 																		2: -1.3862943611198943})

	# def test_xi_matrix_2(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	self.assertEquals(h.xi_matrix(5, s, h.forward(s), h.backward(s))[2], {1: -2.5704875729134073,
	# 																		2: -3.418873166145204,
	# 																		3: -3.8974061320204783,
	# 																		4: -2.080487933135373})

	# def test_forward_backward(self):
	# 	s = "the cat ate the rat"
	# 	p = load_parameters("homework11_simple.pickle")
	# 	h = HMM(p)
	# 	p2 = h.forward_backward(s)
	# 	h2 = HMM(p2)
	# 	self.assertEquals(h2.forward_probability(h2.forward(s)), -34.37400550438377)

	# def test_forward_backward2(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	p2 = h.forward_backward(s)
	# 	h2 = HMM(p2)
	# 	self.assertEquals(h2.forward_probability(h2.forward(s)), -8070.961574771892)

	# def test_update(self):
	# 	s = load_corpus("Brown_sample.txt")
	# 	p = load_parameters("homework11_prob_vector.pickle")
	# 	h = HMM(p)
	# 	h.update(s, 1)
	# 	self.assertEquals(h.forward_probability(h.forward(s)), -7383.3361451482)

if __name__ == '__main__':
    unittest.main()