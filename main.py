from lib.feed_forward import *
import sys
if len(sys.argv) == 2:
    testMethod = sys.argv[1]
    if testMethod == "feedforward":
        ff = FeedForwardNN(500)
        ff.train()
        ff.test()
        ff.plot_input()
    else:
        print "Sorry no such thing is available for test"
