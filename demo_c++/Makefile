CXX = g++
CXX_FLAGS = -Wall -std=c++11 -DHAVE_FFTW -I../sigpack -I/usr/include/eigen3 -lfftw3

%: %.cpp
	@echo '+++++++++++++++++++++++++++++++++++++++++++++'
	@echo 'Compiling $<, making target ./$@ ...'
	@echo '+++++++++++++++++++++++++++++++++++++++++++++'
	$(CXX) $< $(CXX_FLAGS) -o $@
	
	 	
	
	
