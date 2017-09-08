# -*- coding: utf-8 -*-
#
# To 1P3A Warald --- why this code is chosen
#
# This code replicates the Piano Fingering paper by Yonebayashi et al.
# It shows my thinking process of dissecting and reverse-engineering a machine learning algorithm presented in this paper.
# Specifically, it involves number crunching and numerical recipes (such as the calculation of ERFCs).
# The code itself contains training and inference of an HMM model, consisting a full machine learning iteration.
#

import scipy, math, scipy.integrate, sys, random, Queue, threading

# 2015-04-22 First step: Print the estimation E[ln(P(Y|h))] for 
#    two diffferent fingering schemes of the same piece
# 2015-04-23 Used Gradient Descent and found the transition probabilities and Y positions
#    that can replicate the paper's results.

# Debug mode?
DEBUG = False
f_trace = open("last_training.log", "w");

# Finger numbers. Currently only the right hand is considered.
#
#       5 4 3 2 1               1 2 3 4 5
#          .-.                     .-.      
#        .-| |-.                 .-| |-.    
#        | | | |                 | | | |    
#      .-| | | |                 | | | |-.  
#      | | | | |                 | | | | |  
#      | | | | |-.             .-| | | | |  
#      | '     | |             | |     ` |  
#      |       | |             | |       |  
#      |         |             |         |  
#      \         /             \         /  
#        Left hand               Right hand
#
#
states = ('1st', '2nd', '3rd', '4th', '5th')

# The Bounding Boxes of the keys in an octave.
# For some reason when I measured the keyboard I mistakenly started from the F,
#   so the octaves are cut in the middle ..
#
# Y AXIS
#
# ^                        |<---------164 millimetres------>|
# |                        |                                |
# |       Octave N         |          Octave N+1            |   Octave N+2
# |   .---------^--------. .--------------^-----------------. .------^-----.
# |                        
# |   +--+--+--+--+-+--+--+--+--+--+--+--+--+--+--+--+-+--+--+--+--+--+--+--+   ---
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    ^
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |  |  |  | |  |  |  |  |  |  |  |  |  |  |  | |  |  |  |  |  |  |  |    |
# |   |  |#F|  |#G| |#A|  |  |#C|  |#D|  |  |#F|  |#G| |#A|  |  |#C|  |#D|  | 144 millimetres
# |   |  |__|  |__| |__|  |  |__|  |__|  |  |__|  |__| |__|  |  |__|  |__|  |    |
# |   |    |    |    |    |    |    |    |    |    |    |    |    |    |    |    |
# |   | F  | G  | A  | B  | C  | D  | E  | F  | G  | A  | B  | C  | D  | E  |    v
# |   |____|____|____|____|____|____|____|____|____|____|____|____|____|____|   ---
# |
# +-----------------------------------------------------------------------------------------> X AXIS
#
BoundingBoxes0 = { # Format: (Note Name, Octave) = [ List of Bounding Boxes ]
	("F" ,3):  [(0,0,23,50), (0,50,12,144)],
	("#F",3):  [(12,50,26,144)],
	("G" ,3):  [(23,0,47,50),(26,50,39,144)],
	("#G",3):  [(39,50,53,144)],
	("A" ,3):  [(47,0,70,50),(53,50,66,144)],
	("#A",3):  [(66,50,80,144)],
	("B" ,3):  [(70,0,94,50),(80,50,94,144)],
	("C" ,4):  [(94,0,117,50),(94,50,107,144)],
	("#C",4):  [(107,50,122,144)],
	("D", 4):  [(117,0,141,50),(122,50,136,144)],
	("#D",4):  [(136,50,150,144)],
	("E", 4):  [(141,0,164,50),(150,50,164,144)]
}

# 02-13-2015: We found that for white keys, [25, 30, 35, 30, 25] 
#   gives more reasonable result than [15, 15, 15, 15, 15]
#   for (B) in Figure 3
g_Y = [25, 30, 35, 30, 25, 70, 75, 80, 75, 70] # Initial Guess
# After adding Czerny 599 No. 19 First two bars, took 1085 iterations to converge
#g_Y = [29.5651186425, 34.556567213, 46.5894639099, 25.5097678919, 36.4162350794, 70.0, 70.9014258066, 74.5460275605, 75.6177625661, 70.0]
# After adding Czerny 599 No. 19 bars 3 and 4 and first note in bar 5, after 9999 iterations
#g_Y =  [21.9792073765, 33.1052011514, 47.5144223691, 25.7198961808, 35.7647227653, 70.0, 67.2897726961, 73.496131929, 75.617763857, 70.0]
# After another 9999 iterations
#g_Y = [20.6657822004, 33.532796354, 46.5401670595, 25.5777897256, 37.9816310401, 70.0, 67.1747814853, 71.8953747761, 76.3914885222, 70.0 ]
# After Example 9 *DEDUPPED*
#g_Y = [ 24.5187221258, 30.7874549134, 46.8435929038, 27.4452345211, 35.0800942438, 70.0, 68.9214290901, 76.9363534492, 75.2434294104, 70.0 ]

# Expand to more octaves.
BoundingBoxes = dict()

def ComputeBoundingBoxes():
	global BoundingBoxes, BoundingBoxes0
	for o in range(-3, 4):
		delta_x = 164*o
		for k, v in BoundingBoxes0.items():
			k1 = (k[0], k[1]+o); v1 = []
			for bb in v:
				v1.append((bb[0]+delta_x, bb[1], bb[2]+delta_x, bb[3]))
			BoundingBoxes[k1] = v1
ComputeBoundingBoxes()

NoteNameToIdx = {"F":0, "#F":1, "G":2, "#G":3, "A":4, "#A":5, "B":6, "C":7, "#C":8, "D":9, "#D":10, "E":11}
Note3X        = [12,    19,     35,    46,     59,    73,     82,    -58,   -50,    -35,   -21,     -11   ]
NoteIsWhite   = [True,  False,  True,  False,  True,  False,  True,  True,  False,  True,  False,   True  ] 
FingerToIdx   = {'1st':0, '2nd':1, '3rd':2, '4th':3, '5th':4}
def GetContactPoint(no, finger):
	global NoteNameToIdx, g_Y
	idx = NoteNameToIdx[no[0]]
	octave = no[1]
	x = (octave - 3) * 164 + Note3X[idx]
	is_white = NoteIsWhite[idx]
	fidx = FingerToIdx[finger]
	y0_w = g_Y[0:5]
	y0_b = g_Y[5:10]
	if is_white: y = y0_w[fidx]
	else:        y = y0_b[fidx]
	return (x, y)


def Gaussian2D_PDF(x, y, x0, y0, sigma_x_sq, sigma_y_sq):
	xx = (x - x0) * (x - x0); yy = (y - y0) * (y - y0)
	return 1.0 / 2 / 3.1415926 / math.sqrt(sigma_x_sq * sigma_y_sq) \
		* math.exp(-0.5 * (xx / sigma_x_sq + yy / sigma_y_sq))

# tact: tuple (x, y)
# bb:   tuple (x0, y0, x1, y1)
def Gaussian2DProbMass_slow(tact, x0, y0, sigma_x_sq, sigma_y_sq, bbs):
	i1 = 0.0
	for bb in bbs:
		i1 = i1 + scipy.integrate.dblquad(lambda x,y: Gaussian2D_PDF(x,y,x0,y0,sigma_x_sq,sigma_y_sq),\
			(bb[1]-tact[1]), (bb[3]-tact[1]), lambda x: (bb[0]-tact[0]), lambda x: (bb[2]-tact[0]),
			epsabs=1e-3, epsrel=1e-3)[0]
	return i1

# Refer to these pages for how this works
#
# Computing the error function with a computer (when x is not too large)
# http://math.stackexchange.com/a/996598/228339
#
# When x is large, use another asymptotic approximation:
# http://mathworld.wolfram.com/Erf.html (See Formula 18~20)
def erf(x):
	if abs(x) < 20: 
		a = 1.0; b = 1.5;
		ret = 1.0;
		term = 1.0 * a / b * x * x;
		s = 1
		while True:
			ret = ret + term
			term = term * ((a + s) / (b + s) / (s + 1)) * x * x
			if math.isinf(term):
				print x, a, b, s
				assert False
			if abs(term) < 1e-8:
				break
			s = s + 1
		ret = ret * 2 * x / math.sqrt(3.1415926) * math.exp(-x*x)
		return ret
	else:
		term = 1.0 / x;
		ret = 0.0;
		s = 1
		while s < 10:
			ret = ret + term;
			term = term / 2.0 * (s*2-1) / x / x * (-1)
			if abs(term) < 1e-8 or abs(term) > 100: break
			s = s + 1
		ret = 1.0 - math.exp(-x*x) / math.sqrt(3.1415926) * ret
		return ret

def pnorm(x, sigma_x_sq):
	return 0.5 + 0.5 * erf(x / math.sqrt(2 * sigma_x_sq))

def Gaussian2DProbMass(tact, x0, y0, sigma_x_sq, sigma_y_sq, bbs):
	ret = 0.0;
	for bb in bbs:
		# Compute X
		lwr_x = bb[0] - tact[0] - x0
		upr_x = bb[2] - tact[0] - x0
		probmass_x = pnorm(upr_x, sigma_x_sq) - pnorm(lwr_x, sigma_x_sq)
		# Y
		lwr_y = bb[1] - tact[1] - y0
		upr_y = bb[3] - tact[1] - y0
		probmass_y = pnorm(upr_y, sigma_y_sq) - pnorm(lwr_y, sigma_y_sq)

		ret = ret + probmass_x * probmass_y
	return ret
		
	

# If zero then return a very small epsilon.
prob_cache = dict()
def EmissionProbability_safe(state_begin, state_end, note_begin, note_end):
	global prob_cache;
	key = (state_begin, state_end, note_begin, note_end);
#	if prob_cache.has_key(key): ret = prob_cache[key]
	if False: pass
	else: 
		ret = EmissionProbability(state_begin, state_end, note_begin, note_end)
		prob_cache[key] = ret
	EPSILON = 1e-30;
	if ret > EPSILON: return ret
	else: return EPSILON

# Compute Emission Probability Mass
# state_{begin,end}: ['1st', '2nd', '3rd', '4th', '5th']
# note_{begin,end}:  ([#][ABCDEFG], [012345678])
def EmissionProbability(state_begin, state_end, note_begin, note_end):
	global prob_cache;
	end_bbs = BoundingBoxes[note_end];
	start_tact = GetContactPoint(note_begin, state_begin)
	if state_begin == '1st':
		if state_end == '1st': 
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '2nd':
			return 0.81 * Gaussian2DProbMass(start_tact,  42, 25, 900, 30, end_bbs) + \
			       0.19 * Gaussian2DProbMass(start_tact, -23, 25, 400, 30, end_bbs)
		elif state_end == '3rd':
			return 0.89 * Gaussian2DProbMass(start_tact,  50, 30, 900, 30, end_bbs) + \
			       0.11 * Gaussian2DProbMass(start_tact, -16, 30, 100, 30, end_bbs)
		elif state_end == '4th':
			return 0.91 * Gaussian2DProbMass(start_tact,  85, 25, 900, 30, end_bbs) + \
			       0.09 * Gaussian2DProbMass(start_tact, -16, 25, 100, 30, end_bbs)
		elif state_end == '5th':
			return 0.95 * Gaussian2DProbMass(start_tact, 110,  0, 900, 30, end_bbs) + \
			       0.05 * Gaussian2DProbMass(start_tact, -21,  0, 400, 30, end_bbs)
		else: assert(False)
	elif state_begin == '2nd':
		if state_end == '1st': 
			return 0.81 * Gaussian2DProbMass(start_tact, -42, -25, 900, 30, end_bbs) + \
			       0.19 * Gaussian2DProbMass(start_tact,  23, -25, 400, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, 23, 10, 180, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, 50, 0, 200, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 82, -25, 200, 30, end_bbs)
		else: assert(False)
	elif state_begin == '3rd':
		if state_end == '1st':
			return 0.89 * Gaussian2DProbMass(start_tact, -50, -30, 900, 30, end_bbs) + \
			       0.11 * Gaussian2DProbMass(start_tact,  16, -30, 100, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -23, 10, 180, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact,   0,  0, 5,   30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact,  18, -10, 190, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 57, -25, 250, 30, end_bbs)
		else: assert(False)
	elif state_begin == '4th':
		if state_end == '1st':
			return 0.91 * Gaussian2DProbMass(start_tact, -85, -25, 900, 30, end_bbs) + \
			       0.09 * Gaussian2DProbMass(start_tact, 16, -25,  100, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -50, 0, 200, 30, end_bbs)
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, -18, 10, 190, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 20, -20, 200, 30, end_bbs);
		else: assert(False)
	elif state_begin == '5th':
		if state_end == '1st':
			return 0.95 * Gaussian2DProbMass(start_tact, -110, 0, 900, 30, end_bbs) + \
			       0.05 * Gaussian2DProbMass(start_tact,  21,  0, 400, 30, end_bbs)
		elif state_end == '2nd':
			return Gaussian2DProbMass(start_tact, -82, 25, 200, 30, end_bbs);
		elif state_end == '3rd':
			return Gaussian2DProbMass(start_tact, -57, 25, 250, 30, end_bbs)
		elif state_end == '4th':
			return Gaussian2DProbMass(start_tact, -20, 20, 200, 30, end_bbs)
		elif state_end == '5th':
			return Gaussian2DProbMass(start_tact, 0, 0, 5, 30, end_bbs)
		else: assert(False)
	else: assert(False)

# ========================================================================
# Stolen from
# https://en.wikipedia.org/wiki/Viterbi_algorithm
# ========================================================================
def viterbi(obs, states, start_p, trans_p, EmissionProbability):
	print "Observations: %s" % " ".join(["".join([str(xx) for xx in x]) for x in obs])

	V = [{}]
	path = {}
 
	# Initialize base cases (t == 0)
	for y in states:
		V[0][y] = math.log(start_p[y])
		path[y] = [y]
 
	# Run Viterbi for t > 0
	for t in range(1, len(obs)):
		V.append({})
		newpath = {}
 
		for y in states:
			(lg_prob, state) = max(( V[t-1][y0] + \
				math.log(trans_p[y0][y] * \
				EmissionProbability_safe(y0, y, obs[t-1], obs[t]) # 2015-02-12: Added _safe to handle cases when probability is 0
				),\
				y0) for y0 in states)
			V[t][y] = lg_prob
			newpath[y] = path[state] + [y]
 
		# Don't need to remember the old paths
		path = newpath
	n = 0		   # if only one element is observed max is sought in the initialization values
	if len(obs) != 1:
		n = t
	if DEBUG == True: print_dptable(V)
	(lg_prob, state) = max((V[n][y], y) for y in states)
	return (lg_prob, path[state])
 
# Don't study this, it just prints a table of the steps.
def print_dptable(V):
	s = "   " + " ".join(("%7d" % i) for i in range(len(V))) + "\n"
	for y in V[0]:
		s += "%.5s: " % y
		s += " ".join("%.7s" % ("%f" % v[y]) for v in V)
		s += "\n"
	print(s)

#for k in sorted(BoundingBoxes.keys(), key=lambda x:BoundingBoxes[x][0]):
#	print k, BoundingBoxes[k]


# First 9 notes from Bach's Two Part Invention No.1

start_probability = {'1st': 0.20, '2nd': 0.20, 
	'3rd': 0.20, '4th':0.20, '5th': 0.20}
 
# Initial Guess
transition_probability = {
   '1st' : {'1st': 0.20, '2nd': 0.20, '3rd': 0.20, '4th': 0.20, '5th': 0.20},
   '2nd' : {'1st': 0.20, '2nd': 0.20, '3rd': 0.20, '4th': 0.20, '5th': 0.20},
   '3rd' : {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th': 0.16, '5th': 0.21},
   '4th' : {'1st': 0.23, '2nd': 0.23, '3rd': 0.16, '4th': 0.23, '5th': 0.16},
   '5th' : {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th': 0.16, '5th': 0.21},
}

# After adding Czerny 599 No. 19 first two bars, took 1085 iterations
#transition_probability = {
#	'4th': {'4th':0.24797558925860247, '5th':0.1205158637659108 , '2nd':0.15417775950273835, '3rd':0.245671387100048  ,'1st':0.23165940037270047 },
#	'5th': {'4th':0.21279823308551415, '5th':0.19488034021573933, '2nd':0.19488034021573933, '3rd':0.21179215962082762,'1st':0.18564892686217957 },
#	'2nd': {'4th':0.1884235648429344 , '5th':0.21474141074892167, '2nd':0.04382447161631988, '3rd':0.24226303593688173,'1st':0.31074751685494234 },
#	'3rd': {'4th':0.1511174998019772 , '5th':0.21007650523548088, '2nd':0.26826353211668735, '3rd':0.10492470892915555,'1st':0.2656177539166992  },
#	'1st': {'4th':0.15869676067854935, '5th':0.2633225608505323 , '2nd':0.23082580396216898, '3rd':0.2591090481722808 ,'1st':0.08804582633646856 }
#}
# Added bar 3 and 4, and the first note in bar 5
# ... after 9999 iterations
#transition_probability = {
#'1st': {'1st': 0.055724711316745126, '5th': 0.379505720768261  , '4th': 0.17900768913449563, '3rd': 0.2440828906025476 , '2nd': 0.14167898817795058  },
#'5th': {'1st': 0.17925110578630987 , '5th': 0.1881643867815912 , '4th': 0.2291910491745617 , '3rd': 0.2152290714759461 , '2nd': 0.1881643867815912   },
#'4th': {'1st': 0.267239127332642   , '5th': 0.13739176076903903, '4th': 0.28635400032636565, '3rd': 0.20475286713565133, '2nd': 0.10426224443630198  },
#'3rd': {'1st': 0.3958142030389931  , '5th': 0.19420672377592288, '4th': 0.13174227998402147, '3rd': 0.06163923060870999, '2nd': 0.21659756259235255  },
#'2nd': {'1st': 0.3788540983447736  , '5th': 0.14658707071081495, '4th': 0.16115170451852787, '3rd': 0.27609889211883054, '2nd': 0.037308234307053094 }
#}
# After another 9999 iterations
#transition_probability = {
#'2nd':{ '2nd':0.031010486144001228, '5th': 0.12184270871022854, '4th': 0.18405463837028796, '3rd': 0.2820431772297402 , '1st': 0.38104898954574207 },
#'5th':{ '2nd':0.192042422920251   , '5th': 0.192042422920251  , '4th': 0.2540103977360195 , '3rd': 0.17895931571000603, '1st': 0.1829454407134724  },
#'4th':{ '2nd':0.11045288293395325 , '5th': 0.12431612014441144, '4th': 0.3035009737928436 , '3rd': 0.1807418772690803 , '1st': 0.2809881458597114  },
#'3rd':{ '2nd':0.21050771972130017 , '5th': 0.1914344635960726 , '4th': 0.13859078155672325, '3rd': 0.05753527791302578, '1st': 0.4019317572128781  },
#'1st':{ '2nd':0.1338501041154644  , '5th': 0.39120193882860277, '4th': 0.1936138071380788 , '3rd': 0.22697536516352435, '1st': 0.05435878475432976 }
#}

# After training example 9 *DEDUPED* has been fed.
# It seems dupes are the source of evil.
transition_probability = {
	'4th': {'4th':0.259157,'5th':0.166471,'2nd':0.161543,'3rd':0.175367,'1st':0.237462},
	'5th': {'4th':0.183157,'5th':0.211204,'2nd':0.211204,'3rd':0.183232,'1st':0.211204},
	'2nd': {'4th':0.131409,'5th':0.13899,'2nd':0.228213,'3rd':0.232053,'1st':0.269335},
	'3rd': {'4th':0.15366,'5th':0.197049,'2nd':0.186793,'3rd':0.197049,'1st':0.265449},
	'1st': {'4th':0.106405,'5th':0.230476,'2nd':0.20606,'3rd':0.275574,'1st':0.181486}
}

# When the previous note is the same as the current node,
#  use this transition probability table.
transition_probability_rep = {
	
}

# Thumb on C, Middle finger on E, Next up is D and F - can get expected result.
# Thumb on C3, little finger on C4, next up is D3 and D4 - CANNOT get expected result.

if False:
	src_fs = ["1st", "5th"];
	src_notes = [("C", 3), ("C", 4)];
	dest_notes= [("D", 3), ("D", 4)]; # Ascending order
	for i in range(0, len(states)-1):
		for j in range(i, len(states)):
			prob = 1.0;
			dest_f_1 = states[i]; dest_f_2 = states[j];
			dest_fs = [dest_f_1, dest_f_2];
			for idx0 in range(0, 2):
	#			for idx1 in range(0, 2):
				idx1 = idx0;
				prob = prob * EmissionProbability_safe(
					src_fs[idx0], dest_fs[idx1], src_notes[idx0],
					dest_notes[idx1]);
			prob = math.log(prob);
			print "%s -> %s: %f" % (" ".join(src_fs), " ".join(dest_fs),
				prob)
	exit(0);

def do_Permutation(elts, num, stack, last_idx):
	if len(stack) == num:
		yield tuple(stack)
	else:
		for i in range(last_idx, len(elts)):
			e = elts[i]
			if not e in stack:
				for x in do_Permutation(elts, num, stack[:]+[e], i+1):
					yield x
			

def Permutation(elts, num):
	for x in do_Permutation(elts, num, [], 0):
		yield x

def PrintPathDetails(obs, path, start_p, trans_p, emission_probability):
	f_prev = None; notes_prev = None; sum_p = 0;
	for t in range(0, len(obs)):
		f_curr = path[t]
		notes_curr = obs[t]
		ret, tran_p, emi_p, vert_p = ComputeAllTransitionCost(f_prev, f_curr, notes_prev, notes_curr, start_p, trans_p,
			emission_probability);
		sum_p += ret
		if t==0:
			print "[%s] @ %s, Starting, LgProb=%g" % (
				", ".join([str(x) for x in f_curr]), "".join([str(x) for x in obs[t]]),
				tran_p
			)
		else:
			print "[%s] @ %s, LgTransProb=%g, LgEmissionProb=%g, VertP=%g, SumLgProb=%g" % (
				", ".join([str(x) for x in f_curr]), "".join([str(x) for x in obs[t]]),
				tran_p, emi_p, vert_p, sum_p
			)
		notes_prev = notes_curr; f_prev = f_curr;

def GetPathCost(obs, path, start_p, trans_p, emission_probability):
	f_prev = None; notes_prev = None; sum_p = 0;
	for t in range(0, len(obs)):
		f_curr = path[t]
		notes_curr = obs[t]
		ret, _, _, _ = ComputeAllTransitionCost(f_prev, f_curr, notes_prev, notes_curr, start_p, trans_p,
			emission_probability);
		sum_p += ret
		f_prev = f_curr
		notes_prev = notes_curr
	return sum_p

# Called for each new note (or note group)
# If f_prev is None or notes_prev is None, then compute the probability
#   for the first fingering decision
def ComputeAllTransitionCost(f_prev, f_curr, notes_prev, notes_curr, start_p, trans_p, emission_probability):
	ret = 0; tran_p = 0; emi_p = 0; vert_p = 0
	# Part 1: Transition Probability
	if f_prev is None or notes_prev is None:
		for f1 in f_curr:
			tran_p += math.log(start_p[f1])
		ret += tran_p;
	else:
		for nidx1, f1 in enumerate(f_curr):
			for nidx0, f0 in enumerate(f_prev):
				tran_p += math.log(trans_p[f0][f1])
				emi_p += math.log(emission_probability(f0, f1, notes_prev[nidx0], notes_curr[nidx1]))
		ret = ret + tran_p
		ret = ret + emi_p 
	
	# Part 2: Inside-a-chord cost
	if len(f_curr) > 1:
		for i in range(0, len(f_curr)-1):
			vert_p += math.log(trans_p[f_curr[i]][f_curr[i+1]])
	ret = ret + vert_p
	diff = ret - vert_p - emi_p - tran_p
	if abs(diff) > 1e-5: assert(False)
	return ret, tran_p, emi_p, vert_p
		

def viterbiPoly(obs, states, start_p, trans_p, emission_probability):
	Vprev = {} # Key: tuple, states at the previous Time Step
	           # Value: probability and note played
	
	Vcurr = {}

	path = {}
	path_prev = {}

	for y in states:
		Vprev[tuple([y])] = (0, (None))
		path[tuple([y])]  = [([y])]
	
	for t in range(0, len(obs)):
		if DEBUG: print "t=%d" % t
		V = {}
		curr_notes = obs[t]
		if type(curr_notes) == tuple:
			curr_notes = [curr_notes]

		# All possible fingerings of this "unit"
		# (Assuming notes in a chord are in ascending order)
		num_notes = len(curr_notes)

		k0 = sorted(Vprev.keys(), key=lambda x : Vprev[x][0], reverse=True)[0]
		if DEBUG: print k0, Vprev[k0][1]

		for fingers in Permutation(states, num_notes):
			max_lg_prob = -1e20
			for fingers_prev, prob_and_notes_prev in Vprev.items():
				delta_lg_prob = 0

				lg_prob_prev  = prob_and_notes_prev[0]
				notes_prev = prob_and_notes_prev[1]

				delta_lg_prob, _, _, _ = ComputeAllTransitionCost(fingers_prev, fingers,
					notes_prev, curr_notes, start_p, trans_p, emission_probability);

				if lg_prob_prev + delta_lg_prob > max_lg_prob:
					Vcurr[fingers] = (lg_prob_prev + delta_lg_prob, tuple(curr_notes))
					if t > 0: path[fingers] = path_prev[fingers_prev]
					else: path[fingers] = []
					max_lg_prob = lg_prob_prev + delta_lg_prob
					if DEBUG: print "%s %s ---> %s %s = %g" % (
						str(fingers_prev), str(notes_prev),
						str(fingers),      str(curr_notes),
						lg_prob_prev + delta_lg_prob)

		Vprev = Vcurr; Vcurr = {}
		for k, v in path.items():
			path_prev[k] = v + [k]
		path = {}
	k0 = sorted(Vprev.keys(), key=lambda x : Vprev[x][0], reverse=True)[0]
	the_path = path_prev[k0]
	return the_path, Vprev[k0][0]


##############################################################################
## Main Program!
##############################################################################

def GetFingeringDiff(x, y):
	assert len(x) == len(y)
	diff = []
	for idx in range(0, len(x)):
		xx = x[idx]; yy = y[idx];
		if type(xx) == str:
			if type(yy) == tuple or type(yy) == list:
				if len(yy) == 1 and yy[0] == xx: 
					diff.append(None)
				else: 
					diff.append((xx, yy[0]))
				continue
			elif type(yy) == str:
				if xx == yy:
					diff.append(None)
				else:
					diff.append((xx, yy));
				continue
			else:
				assert False
		elif type(xx) == list or type(xx) == tuple:
			if type(yy) == str:
				if len(xx) == 1 and xx[0] == yy:
					diff.append(None)
				else:
					diff.append((xx[0], yy))
				continue
			else:
				if len(xx) != len(yy):
					diff.append((xx, yy));
				else:
					same = True
					for idx1 in range(0, len(xx)):
						if xx[idx1] == yy[idx1]: continue
						else:
							diff.append((xx, yy));
							same = False;
							break;
					if same:
						diff.append(None)
		else:
			diff.append((xx, yy))
	assert len(diff) == len(x)
	return diff


def IsFingeringIdentical(x, y):
	if len(x) != len(y):
		return False
	for idx in range(0, len(x)):
		xx = x[idx]; yy = y[idx];
		if type(xx) == str:
			if type(yy) == tuple or type(yy) == list:
				if len(yy) == 1 and yy[0] == xx: continue
				else: 
					return False
			elif type(yy) == str:
				if xx == yy: continue
				else: 
					return False
		elif type(xx) == list or type(xx) == tuple:
			if type(yy) == str:
				if len(xx) == 1 and xx[0] == yy: continue
				else:
					return False
			else:
				if len(xx) != len(yy):
					return False
				else:	
					for idx1 in range(0, len(xx)):
						if xx[idx1] == yy[idx1]: continue
						else: 
							return False
		else:
			return False
	return True

start_p1 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations1 = [
	# Measure 1
    [("C",3)],[("D",3)],[("E",3)],[("F",3)],[("D",3)],[("E",3)],[("C",3)], \
	[("G",3)],[("C",4)],[("B",3)],[("C",4)],[("B",3)],[("C",4)],
	# Measure 2 
	[("D",4)],[("G",3)],[("A",3)],[("B",3)],[("C",4)],[("A",3)],[("B",3)],[("G",3)],[("D",4)],[("G",4)] ]
y0_1 = [ ["1st"], ["2nd"], ["3rd"], ["4th"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["4th"], ["3rd"], ["4th"], 
         ["3rd"], ["4th"], \
		 ["5th"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["2nd"], ["3rd"], ["1st"], ["3rd"], ["5th"] ]

start_p2 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.21} # Somewhere in the middle so we can set the probabilites
observations2 = [
	[("A",4)], [("G",4)], [("F",4)], [("E",4)], [("D",4)], [("C",4)], [("E",4)], [("D",4)], [("F",4)]
]
y0_2 = [ ["5th"], ["4th"], ["3rd"], ["2nd"], ["1st"], ["2nd"], ["4th"], ["3rd"], ["5th"] ]

start_p3 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations3 = [ [("G",3)], [("#D",4)], [("D",4)], [("#D",4)], [("#A",3)], [("C",4)], [("#D",4)], [("#G",4)] ]
fingers3_me = [ ["1st"],   ["3rd"],    ["1st"],   ["4th"] ]
fingers3_x  = [ ["1st"],   ["4th"],    ["3rd"],   ["4th"] ]
y0_3 = [ ["1st"], ["4th"], ["3rd"], ["4th"], ["2nd"], ["1st"], ["3rd"], ["5th"] ]

start_p4 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations4 = [
	[("C",4)], [("#A",3)],[("#G",3)],[("G",3)],[("#F",3)],[("#D",4)],[("D",4)],[("C",4)],
	[("#A",3)],[("A",3)],[("G",3)],[("F",3)] ]
fingers4_me = [ ["5th"],  ["3rd"],  ["2nd"],  ["1st"] ]
fingers4_x  = [ ["5th"],  ["4th"],  ["3rd"],  ["1st"] ]
y0_4 = [ ['5th'], ['4th'], ['3rd'], ['1st'], ['2nd'], ['5th'], ['4th'], ['3rd'], ['2nd'], ['1st'],
         ['3rd'], ['2nd'] ]

start_p5 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations5 = [
	[("G",3)], [("G",3)], [("A",3)], [("B",3)]
]
y0_5 = [ ["2nd"], ["2nd"], ["3rd"], ["4th"] ]

start_p6 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations6 = [
	[("#G",3)], [("F",4)], [("E",4)], [("D",4)], [("C",4)], [("D",4)], [("C",4)],
	[("B",3)],  [("A",3)],
	[("A",3)],  [("A",4)], [("G",4)],  [("F",4)], [("E",4)], [("G",4)], [("F",4)], [("A",4)], [("G",4)]
]
y0_6 = [ ["1st"], ["5th"], ["4th"], ["3rd"], ["2nd"], ["4th"], ["3rd"],
         ["2nd"], ["1st"], ["1st"], ["5th"], ["4th"], ["3rd"], ["2nd"], ["4th"], ["3rd"], ["5th"], ["4th"] ]

start_p7 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations7 = [
	[("A",3)], [("D",3)], [("C",4)], [("B",3)], [("C",4)], [("C",4)], [("D",4)],
	[("B",3)], [("A",3)], [("G",3)], [("#F",3)], [("E",3)], [("G",3)], [("#F",3)],
	[("A",3)], [("G",3)], [("B",3)], [("A",3)], [("C",4)], [("B",3)], [("D",4)],
	[("C",4)], [("E",4)], [("D",4)], [("B",3)], [("C",4)]
]
y0_7 = [ ["4th"], ["1st"], ["4th"], ["3rd"], ["4th"], ["4th"], ["5th"],
         ["3rd"], ["2nd"], ["1st"], ["3rd"], ["2nd"], ["3rd"], ["2nd"], ["3rd"],
		 ["2nd"], ["3rd"], ["2nd"], ["3rd"], ["2nd"], ["4th"], ["3rd"], ["5th"],
		 ["4th"], ["2nd"], ["3rd"] ]

# Czerny 599 No. 19
# Measures 1 and 2
start_p8 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations8 = [
	[("C",5)], [("D",5)], [("E",5)], [("F",5)], [("G",5)], [("A",5)], [("B",5)], [("C",6)],
	[("A",5)], [("F",5)], [("C",6)], [("A",5)], [("G",5)], [("G",5)]
]
y0_8 = [ ["1st"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["5th"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"], ["1st"] ]

# 去掉了最后那个重复的
# 我猜如果去掉了两个599用例中重复的部分导致能练出来，而留着重复的部分就练不出来的话，blah
start_p8 = {'1st': 0.21, '2nd': 0.21, '3rd': 0.21, '4th':0.21, '5th': 0.14}
observations8 = [
	[("C",5)], [("D",5)], [("E",5)], [("F",5)], [("G",5)], [("A",5)], [("B",5)], [("C",6)],
	[("A",5)], [("F",5)], [("C",6)], [("A",5)], [("G",5)]
]
y0_8 = [ ["1st"], ["2nd"], ["3rd"], ["1st"], ["2nd"], ["3rd"], ["4th"], ["5th"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"] ]

# Measures 3 and 4, plus the 1st note in Measure 5
# Force starting with 1
start_p9 = {'1st': 0.01, '2nd': 0.01, '3rd': 0.01, '4th':0.01, '5th': 1.00}
observations9_dedup = [
	[("C",6)], [("B",5)], [("A",5)], [("G",5)], [("F",5)], [("E",5)], [("D",5)], [("C",5)],
	[("E",5)], [("C",5)], [("G",5)], [("E",5)], [("D",5)]
]
y0_9_dedup = [ ["5th"], ["4th"], ["3rd"], ["2nd"], ["1st"], ["3rd"], ["2nd"], ["1st"],
         ["3rd"], ["1st"], ["5th"], ["3rd"], ["2nd"] ]

# Cannot 
start_p10 = {'1st': 1.00, '2nd': 0.01, '3rd': 0.01, '4th':0.01, '5th': 0.01}
observations10 = [
	[("D",5)], [("E",5)], [("D",5)], [("E",5)], [("F",5)], [("D",5)], [("G",5)], [("F",5)],
	[("E",5)], [("F",5)], [("E",5)], [("F",5)], [("G",5)], [("C",6)], [("G",5)], [("E",5)]
]
y0_10 = [
	["1st"], ["2nd"], ["1st"], ["2nd"], ["3rd"], ["1st"], ["4th"], ["3rd"],
	["1st"], ["2nd"], ["1st"], ["2nd"], ["3rd"], ["5th"], ["3rd"], ["2nd"]
]

# y0 is the training response!
def ProcessTrainingData(obs, states, start_p, trans_p, emission_probability, y0):
	assert len(obs) == len(y0)
	path, y = viterbiPoly(obs,
				    states,
				    start_p,
				    trans_p,
				    EmissionProbability_safe)
	if IsFingeringIdentical(path, y0):
		print "Consistent with training set!"
	else:
		print "Not Consistent with training set!"
		print "Training:", y0
		print "Hyp:     ", path

def viterbiPoly_threadstart(obs, states, start_p, trans_p, emission_prob, output):
	path, cost = viterbiPoly(obs, states, start_p, trans_p, emission_prob)
	output["path"] = path; output["cost1"] = cost;
	print "ok"

def TuningProblem1(training_obs, training_fs, training_starts):

	# Initialize Offender List
	offenders = []
	for i in range(0, len(training_obs)):
		ob = training_obs[i]
		the_list = []
		for j in range(0, len(ob)):
			the_list.append(dict())
		offenders.append(the_list)

	global g_Y
	Xname = ["WY1", "WY2", "WY3", "WY4", "WY5", "BY1", "BY2", "BY3", "BY4", "BY5"]
	gradients = [0] * 10
	gradients_t = {
	   '1st' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '2nd' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '3rd' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '4th' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	   '5th' : {'1st': 0, '2nd': 0, '3rd': 0, '4th': 0, '5th': 0},
	}
	global prob_cache
	learning_rate = 1
	learning_rate_t = 0.0001

	# Print Column
	line = []
	line.append(">>> iter, ")
	for i in range(0, len(training_obs)):
		line.append("diff%d, " % i)
	for i in range(0, len(g_Y)):
		line.append("ypos%d, " % i)
	keys0 = ['1st', '2nd', '3rd', '4th', '5th']
	for k0 in range(0, len(keys0)):
		for k1 in range(0, len(keys0)):
			line.append("tprob%d%d, " % (k0, k1))
	line.append("\n")
	sys.stdout.write(''.join(line))
	f_trace.write((''.join(line))[4:])
	sys.stdout.flush()

	for iter in range(0, 99999):
		ok = True
		is_error = []
		wrong_paths = []
		diffs = []

		scratch = []
		for i in range(0, len(training_obs)):
			scratch.append(dict())
		threads = []

		for tidx in range(0, len(training_obs)):
			path, cost1 = viterbiPoly(training_obs[tidx],
				states,
				training_starts[tidx],
				transition_probability,
				EmissionProbability_safe)
			if not IsFingeringIdentical(path, training_fs[tidx]):
				cost2 = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				diff = GetFingeringDiff(training_fs[tidx], path);
				for i, x in enumerate(diff):
					if x is None: continue
					else:
						assert type(x) == tuple
						key0 = tuple(x[0])  # Fingering in training data
						val0 = tuple(x[1])  # Fingering generated using VS
						if not offenders[tidx][i].has_key(val0):
							offenders[tidx][i][val0] = 0
						offenders[tidx][i][val0] += 1
					
				print "Training example %d, not eq training fingering, cost: %g vs %g, diff=%g" %  \
					(tidx, cost1, cost2, cost2-cost1)
				diffs.append(cost2-cost1)
				wrong_paths.append(path)
				is_error.append(True)
				ok = False
			else:
				print "Training example %d,     eq training fingering, cost: %g" %  (tidx, cost1)
				is_error.append(False)
				wrong_paths.append(None)
				diffs.append(0)

		if ok: break

		# Compute Derivatives
		delta = 0.01

		for x in range(0, len(g_Y)):
			gradients[x] = 0;
		for v in gradients_t.values():
			for vk in v.keys():
				v[vk] = 0

		# Choose one offending data point for stochastic gradient descent
		chosen_idx = -1
		while chosen_idx == -1:
			x = random.randint(0, len(training_obs)-1)
			if is_error[x] == True:
				chosen_idx = x
				break

		# Derivative to Y position of contact points
		for tidx in [chosen_idx]:
			sys.stderr.write(str(tidx))
			if is_error[tidx] == False: continue
			for x in range(0, len(g_Y)):
				sys.stderr.write(".")
				xold = g_Y[x]
				cost_f_hat  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				cost_f_gold = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)

				g_Y[x] = xold + delta
				cost_f_hat_1  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)
				cost_f_gold_1 = GetPathCost(training_obs[tidx], training_fs[tidx],
					training_starts[tidx], transition_probability, EmissionProbability_safe)

				dd_dx = ((cost_f_gold_1 - cost_f_hat_1) - (cost_f_gold - cost_f_hat)) / delta;
				g_Y[x] = xold;
				gradients[x] += dd_dx

		delta = 0.003
		# Derivative to Transition Probabilities
		for tidx in [chosen_idx]:
			sys.stderr.write(str(tidx))
			if is_error[tidx] == False: continue
			for k, v in gradients_t.items():
				for vk in v.keys():
					sys.stderr.write(".")
					oldv = transition_probability[k][vk]

					cost_f_hat  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)
					cost_f_gold = GetPathCost(training_obs[tidx], training_fs[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)

					transition_probability[k][vk] += delta

					cost_f_hat_1  = GetPathCost(training_obs[tidx], wrong_paths[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)
					cost_f_gold_1 = GetPathCost(training_obs[tidx], training_fs[tidx],
						training_starts[tidx], transition_probability, EmissionProbability_safe)

					dd_dx = ((cost_f_gold_1 - cost_f_hat_1) - (cost_f_gold - cost_f_hat)) / delta;
					transition_probability[k][vk] = oldv
					gradients_t[k][vk] += dd_dx

		for x in range(0, len(g_Y)):
			g_Y[x] += gradients[x] * learning_rate
		for k, v in gradients_t.items():
			for vk in v.keys():
				transition_probability[k][vk] += gradients_t[k][vk] * learning_rate_t
			sum1 = sum(transition_probability[k].values())
			for vk in v.keys():
				transition_probability[k][vk] /= sum1
		sys.stderr.write("\n")
#		print "Gradients of Y :     ", ", ".join([str(x) for x in gradients])
#		print "Gradients of Tran Prob:"
#		for k, v in gradients_t.items():
#			print k, ", ".join([str(x) for x in v.items()])
		print "New Y: ", ", ".join([str(x) for x in g_Y])
		print "New Transition Prob: "
		for k, v in transition_probability.items():
			sys.stdout.write("'%s': {" % k)
			for k1, v1 in v.items():
				sys.stdout.write("'%s':%g," % (k1, v1))
			sys.stdout.write("}\n")
		line = []
		line.append(">>> %d, " % iter)
		line.append(", ".join([str(x) for x in diffs]))
		line.append(", ")
		line.append(", ".join([str(x) for x in g_Y]))
		for k0 in keys0:
			for k1 in keys0:
				line.append(", ")
				line.append(str(transition_probability[k0][k1]))
		line.append("\n")
		sys.stdout.write(''.join(line))
		f_trace.write((''.join(line))[4:])
		sys.stdout.flush()

		# Print Offenders
		counts = []
		print("==========Offenders list==============")
		for tidx, o in enumerate(offenders):
			sum0 = 0
			for ety in o:
				sum0 = sum0 + sum(ety.values())
			print "Training example %d has %d offenders" % (tidx, sum0)
			counts.append(sum0)
		print("==========Detailed Offenders list=====")
		for tidx, o in enumerate(offenders):
			if counts[tidx] > 0:
				print "Training example %d (%d):" % (tidx, counts[tidx])
				for i, ety in enumerate(o):
					if ety is not None:
						correct = training_fs[tidx][i]
						for i1, kv in enumerate(ety.items()):
							print "  Note [%d], %s, %s->%s (%d)" % (i, \
								str(training_obs[tidx][i]), correct, kv[0], kv[1])
			else:
				print "There are no offenders for training example %d." % tidx


def TestKumarsEquation():
	tact = [1,2]; x0=1; y0=2; sigma_x_sq=300; sigma_y_sq=90;
	bbs = [ [-55,-10,34,-5] ]
	print Gaussian2DProbMass(tact, x0, y0, sigma_x_sq, sigma_y_sq, bbs)
	print Gaussian2DProbMass_fast(tact, x0, y0, sigma_x_sq, sigma_y_sq, bbs)

# I think it passed  (2015-04-24)
#TestKumarsEquation(); 
#exit(0)

TuningProblem1(
	[
		observations1, observations2, observations3, 
		observations4, observations5, observations6, 
		observations7,# observations8, observations9_dedup,
		#observations10
	], 
	[y0_1, y0_2, y0_3, y0_4, y0_5, y0_6, y0_7,],# y0_8, y0_9_dedup, y0_10],
	[start_p1, start_p2, start_p3, start_p4, start_p5, start_p6, start_p7,]# start_p8, start_p9, start_p10]
)
exit(0)

# PARAMETER TUNING FOR TRAINING EXAMPLES IN IJCAI07

# A
print "\n(A) in Figure 3 in the paper"
ProcessTrainingData(observations1,
			   states,
			   start_p1,
			   transition_probability,
			   EmissionProbability,
			   y0_1)

# (B) in Figure 3
print "\n(B) in Figure 3 in the paper"
ProcessTrainingData(observations2,
				   states,
				   start_p2,
				   transition_probability,
				   EmissionProbability,
				   y0_2)

# C, first half
print "\nLeft half of (C) in Figure 3 in the paper"

ProcessTrainingData(observations3,
			   states,
			   start_p3,
			   transition_probability,
			   EmissionProbability_safe,
			   y0_3)

# C, second half
print "\nRight half of (C) in Figure 3 in the paper"

ProcessTrainingData(observations4,
			   states,
			   start_p4,
			   transition_probability,
			   EmissionProbability_safe,
			   y0_4)

# D, first half
print "\nFirst half of (D) in Figure 3 in the paper"
ProcessTrainingData(observations5,
				   states,
				   start_p5,
				   transition_probability,
				   EmissionProbability,
				   y0_5)

# D, second half
print "\nSecond half of (D) in Figure 3 in the paper"
ProcessTrainingData(observations6,
				   states,
				   start_p6,
				   transition_probability,
				   EmissionProbability,
				   y0_6)

# E
print "\n(E) in Figure 3 in the paper"
ProcessTrainingData(observations7,
			   states,
			   start_p7,
			   transition_probability,
			   EmissionProbability,
			   y0_7)

exit(0)

if False:
	test1 = [ ("C", 3), [("C",3), ("E",3) ], [("D",3), ("F",3)], [("E",3), ("G",3)] ]
	print viterbiPoly(test1,
				   states,
					{'1st': 0.22, '2nd': 0.21, 
						'3rd': 0.21, '4th':0.21, '5th': 0.14},
				   transition_probability,
				   EmissionProbability_safe)

# sugarcoated haws on a stick.
if False:
	x = [ [("B",4),("E",5),("#G",5),("B",5)], [("#C",5),("E",5),("A",5),("#C",6)] ]
	start_p = {'1st': 100, '2nd': 0.11, '3rd': 0.11, '4th':0.11, '5th': 0.11}
	path = viterbiPoly(x,
				   states,
				   start_p,
				   transition_probability,
				   EmissionProbability_safe)
	PrintPathDetails(x, path, start_p, transition_probability, EmissionProbability_safe)
	alt_path = [ ['1st', '2nd', '3rd', '5th'], ['1st', '2nd', '4th', '5th'] ]
	PrintPathDetails(x, alt_path, start_p, transition_probability, EmissionProbability_safe)


# Test case from AIDA 07
aida1 = [ ("A", 3), ("#A", 3), ("C", 4), ("D", 4), ("C", 4), [("A",3), ("F",4)],
	[("C",4),("G",4)], [("F",4),("A",4)], 
	("A",3),("#A",3),("C",4),("D",4),("C",4), [("A",3),("C",4),("F",4)],
		[("#A",3),("C",4),("E",4)], [("A",3),("C",4),("F",4)],
		
		]
print viterbiPoly(aida1,
			   states,
				{'1st': 100, '2nd': 0.11, 
					'3rd': 0.11, '4th':0.11, '5th': 0.11},
			   transition_probability,
			   EmissionProbability_safe)

aida2 = [[("E",4),("G",4)], [("F",4),("#G",4)], [("#F",4),("A",4)], [("G",4),("#A",4)],
	[("#A",4),("D",5)], [("A",4),("C",5)], [("G",4),("#A",4)], [("F",4),("A",4)] ]
print viterbiPoly(aida2,
			   states,
				{'1st': 100, '2nd': 0.11, 
					'3rd': 0.11, '4th':0.11, '5th': 0.11},
			   transition_probability,
			   EmissionProbability_safe)

aida3 = [ [("E",4),("G",4)], [("F",4),("G",4),("B",4)], [("E",4),("G",4),("C",5)],
	[("E",4),("G",4)], [("F",4),("#G",4)], [("#F",4),("A",4)], [("G",4),("#A",4)],
	[("#A",4),("D",5)], [("A",4),("C",5)], [("G",4),("#A",4)], [("F",4),("A",4)],
	[("E",4),("G",4)], ("C",4), ("B",3), ("#A",3), ("A",3), ("#A",3), ("C",4), ("D",4),
	("C",4), [("A",3),("F",4)], [("C",4),("G",4)], [("F",4),("A",4)] ]
print viterbiPoly(aida3,
			   states,
				{'1st': 100, '2nd': 0.11, 
					'3rd': 0.11, '4th':0.11, '5th': 0.11},
			   transition_probability,
			   EmissionProbability_safe)

exit(0)


print "\nExtreme Case 1"
scale1 = [ ("C", 3), ("C", 4), ("D", 3), ("D", 4), ("E", 3), ("E", 4) ]
print viterbi(scale1,
			   states,
				{'1st': 0.21, '2nd': 0.21, 
					'3rd': 0.21, '4th':0.21, '5th': 0.14},
			   transition_probability,
			   EmissionProbability)

print "Chord 1"
chord1 = [ ("C", 3), ("E", 3), ("G", 3) ]
print viterbi(chord1,
			   states,
				{'1st': 0.21, '2nd': 0.21, 
					'3rd': 0.21, '4th':0.21, '5th': 0.21},
			   transition_probability,
			   EmissionProbability)

print "\nC Major Scale x 3 octaves"
scale1 = [ 
	("C",3),("D",3),("E",3),("F",3),("G",3),("A",3),("B",3),
	("C",4),("D",4),("E",4),("F",4),("G",4),("A",4),("B",4),
	("C",5),("D",5),("E",5),("F",5),("G",5),("A",5),("B",5),
];
print viterbi(scale1,
			   states,
				{'1st': 0.21, '2nd': 0.21, 
					'3rd': 0.21, '4th':0.21, '5th': 0.14},
			   transition_probability,
			   EmissionProbability)

print "\nA Minor Scale x 3 octaves"
scale = [
	("A",2), ("B",2), ("C",3), ("D",3), ("E",3), ("F",3), ("#G",3),
	("A",3), ("B",3), ("C",4), ("D",4), ("E",4), ("F",4), ("#G",4),
	("A",4), ("B",4), ("C",5), ("D",5), ("E",5), ("F",5), ("#G",5),
]
print viterbi(scale,
			   states,
				{'1st': 0.21, '2nd': 0.21, 
					'3rd': 0.21, '4th':0.21, '5th': 0.14},
			   transition_probability,
			   EmissionProbability)

print "\nF Major Scale x 3 octaves"
scale = [
	("F",2), ("G",2), ("A",2), ("#A",2), ("C",3), ("D",3), ("E",3),
	("F",3), ("G",3), ("A",3), ("#A",3), ("C",4), ("D",4), ("E",4),
	("F",4), ("G",4), ("A",4), ("#A",4), ("C",5), ("D",5), ("E",5),
]
print viterbi(scale,
			   states,
				{'1st': 0.21, '2nd': 0.21, 
					'3rd': 0.21, '4th':0.21, '5th': 0.14},
			   transition_probability,
			   EmissionProbability)

print "\nB-flat Major Scale x 3 octaves"
scale2 = [ 
	("#A",2),("C",3),("D",3),("#D",3),("F",3),("G",3),("A",3),
	("#A",3),("C",4),("D",4),("#D",4),("F",4),("G",4),("A",4),
	("#A",4),("C",5),("D",5),("#D",5),("F",5),("G",5),("A",5),
];

if False:
	print viterbi(scale2,
				   states,
					{'1st': 0.21, '2nd': 0.21, 
						'3rd': 0.21, '4th':0.21, '5th': 0.14},
				   transition_probability,
				   EmissionProbability)


#print EmissionProbability("1st", "4th", ("G",3), ("E",3)) # 0.014

#print EmissionProbability("3rd", "4th", ("F",3), ("E",3)) # 0.014
#print EmissionProbability("3rd", "2nd", ("F",3), ("E",3)) # 0.60
#print EmissionProbability("1st", "2nd", ("F",3), ("E",3)) # 0.10
